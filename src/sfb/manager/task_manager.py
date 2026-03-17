import torch


def _corrupt_targets(y: torch.Tensor, vocab_size: int, noise_rate: float) -> torch.Tensor:
    """Corrupt a fraction of target tokens with random wrong labels (train-time only)."""
    if noise_rate <= 0 or vocab_size <= 1:
        return y
    mask = torch.rand_like(y, dtype=torch.float32, device=y.device) < noise_rate
    wrong = torch.randint(0, vocab_size, y.shape, device=y.device, dtype=y.dtype)
    wrong = torch.where(wrong == y, (y + 1) % vocab_size, wrong)  # ensure wrong != y
    return torch.where(mask, wrong, y)


class TaskManager:
    def __init__(self, task, model, optimizer, device, target_noise: float = 0.0):
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.target_noise = float(target_noise)
        self.vocab_size = getattr(task.generator, "vocab_size", None)
        self.history = []

    def train_step(self, batch_size: int):
        self.model.train()
        x, y = self.task.get_batch(batch_size, split="train")
        x, y = x.to(self.device), y.to(self.device)
        if self.target_noise > 0 and self.vocab_size is not None:
            y = _corrupt_targets(y, self.vocab_size, self.target_noise)

        logits = self.model(x)
        loss = self.task.loss_fn(logits, y)
        train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), train_acc

    def eval_step(self, batch_size: int):
        self.model.eval()
        with torch.no_grad():
            x, y = self.task.get_batch(batch_size, split="eval")
            x, y = x.to(self.device), y.to(self.device)
            val_loss = self.task.loss(self.model, (x, y)).item()
            val_acc = self.task.evaluate(self.model, (x, y))
            return val_loss, val_acc

    def train(
        self,
        n_steps: int,
        batch_size: int = 64,
        eval_every: int = 1000,
        early_stopping: bool = True,
        patience: int = 3,
        no_hope_threshold: float = 0.01,
        no_hope_after_evals: int = 1,
        no_hope_first_eval_at: int | None = 100,
    ):
        """
        Train for up to n_steps. Returns (history, early_stopped, stopped_at_step).
        Early stopping: stop if no val_acc improvement for `patience` evals, or if
        val_acc < no_hope_threshold after `no_hope_after_evals` evals.
        If no_hope_first_eval_at is set (e.g. 100), run one eval at that step and
        stop immediately if val_acc < no_hope_threshold (saves time on hopeless runs).
        """
        self.history = []
        early_stopped = False
        stopped_at_step = n_steps
        best_val_acc = -1.0
        best_state = None  # model state_dict at best val_acc (so we can restore after early stop)
        evals_without_improvement = 0
        min_steps = 2 * eval_every  # don't stop before we have at least 2 evals

        for step in range(n_steps):
            train_loss, train_acc = self.train_step(batch_size)
            current_step = step + 1

            # Optional early eval just for no-hope check (e.g. at step 100)
            if (
                early_stopping
                and no_hope_first_eval_at is not None
                and no_hope_first_eval_at > 0
                and current_step == no_hope_first_eval_at
                and current_step % eval_every != 0
            ):
                val_loss, val_acc = self.eval_step(batch_size)
                self.history.append({
                    "step": current_step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
                print(
                    f"Step {current_step}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
                if val_acc < no_hope_threshold:
                    early_stopped = True
                    stopped_at_step = current_step
                    print(f"  Early stop (no hope at step {current_step}: val_acc={val_acc:.4f} < {no_hope_threshold})")
                    break

            if current_step % eval_every == 0:
                val_loss, val_acc = self.eval_step(batch_size)
                self.history.append({
                    "step": current_step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
                print(
                    f"Step {current_step}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

                if early_stopping:
                    # No hope: still near-zero acc after a few evals
                    num_evals = len(self.history)
                    if num_evals >= no_hope_after_evals and val_acc < no_hope_threshold:
                        early_stopped = True
                        stopped_at_step = current_step
                        print(f"  Early stop (no hope: val_acc={val_acc:.4f} < {no_hope_threshold} after {num_evals} evals)")
                        break
                    # No improvement for patience evals
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        evals_without_improvement = 0
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    else:
                        evals_without_improvement += 1
                    if current_step >= min_steps and evals_without_improvement >= patience:
                        early_stopped = True
                        stopped_at_step = current_step
                        if best_state is not None:
                            self.model.load_state_dict(best_state, strict=True)
                            # move back to device (best_state was copied to CPU)
                            self.model.to(self.device)
                        print(f"  Early stop (no improvement for {patience} evals, best val_acc={best_val_acc:.4f})")
                        break

        # Final eval if we never hit eval_every or didn't eval at n_steps
        if not early_stopped:
            if self.history and self.history[-1]["step"] != n_steps:
                val_loss, val_acc = self.eval_step(batch_size)
                self.history.append({
                    "step": n_steps,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
                print(
                    f"Step {n_steps}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            elif not self.history:
                val_loss, val_acc = self.eval_step(batch_size)
                self.history.append({
                    "step": n_steps,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
                print(
                    f"Step {n_steps}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

        return self.history, early_stopped, stopped_at_step

    def show_examples(self, n_examples=5):
        """Show some test examples with input and predictions"""
        self.model.eval()
        with torch.no_grad():
            x, y = self.task.get_batch(n_examples, split="eval")
            x, y = x.to(self.device), y.to(self.device)
            
            logits = self.model(x)
            preds = logits.argmax(dim=-1)
            
            x = x.cpu()
            y = y.cpu()
            preds = preds.cpu()
            
            print("\nExamples:")
            for i in range(n_examples):
                print(f"\nExample {i+1}:")
                print(f"  Input:  {x[i].tolist()}")
                print(f"  Target: {y[i].tolist()}")
                print(f"  Pred:   {preds[i].tolist()}")
                correct = (preds[i] == y[i]).sum().item()
                print(f"  Correct: {correct}/{len(y[i])}")