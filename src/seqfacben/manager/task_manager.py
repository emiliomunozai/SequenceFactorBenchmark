import torch
import pandas as pd

class TaskManager:
    def __init__(self, task, model, optimizer, device):
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history = []

    def train_step(self, batch_size: int):
        self.model.train()
        x, y = self.task.get_batch(batch_size, split="train")
        x, y = x.to(self.device), y.to(self.device)

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

    def train(self, n_steps: int, batch_size: int = 64, eval_every: int = 1000):
        self.history = []
        for step in range(n_steps):
            train_loss, train_acc = self.train_step(batch_size)

            if (step + 1) % eval_every == 0:
                val_loss, val_acc = self.eval_step(batch_size)
                self.history.append({
                    'step': step + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                })
                print(
                    f"Step {step + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
        return self.history

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