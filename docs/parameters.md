# Controllable Parameters per Factor

This document defines the **min**, **max**, and **step** (or resolution) for each controllable parameter used in data generation and evaluation. Configs and scripts should reference these ranges for sweeps and reproducibility.

---

## Scale

| Parameter         | Min  | Max   | Step / Notes   | Default |
|-------------------|------|-------|----------------|---------|
| `sequence_length` | 16   | 4096  | Power-of-2 or linear step (e.g. 16, 32, 64, …) | 128 |
| `vocabulary_size` | 32   | 65536 | Power-of-2 recommended       | 1024 |
| `batch_size`      | 1    | 256   | 1, 2, 4, 8, …               | 32   |

---

## Breadth

| Parameter           | Min | Max  | Step / Notes   | Default |
|---------------------|-----|------|----------------|---------|
| `task_family_count` | 1   | 16   | 1              | 4       |
| `num_classes`       | 2   | 1024 | Power-of-2 or 1 | 10      |
| `num_operations`    | 1   | 32   | 1              | 8       |

---

## Structure

| Parameter            | Min | Max  | Step / Notes | Default |
|----------------------|-----|------|--------------|---------|
| `nesting_depth`      | 1   | 16   | 1            | 4       |
| `dependency_span`    | 1   | 512  | Power-of-2 or linear | 64  |
| `grammar_complexity` | 2   | 64   | 1            | 8       |

---

## Noise

| Parameter         | Min  | Max  | Step / Notes | Default |
|-------------------|------|------|--------------|---------|
| `input_noise_rate` | 0.0 | 0.5  | 0.05         | 0.0     |
| `label_noise_rate` | 0.0 | 0.3  | 0.05         | 0.0     |
| `eval_dropout`     | 0.0 | 0.2  | 0.02         | 0.0     |

---

## Suggested Sweep Grids (Examples)

- **Scale sweep:** `sequence_length ∈ {64, 128, 256, 512, 1024}`, fix others at default.
- **Noise sweep:** `input_noise_rate ∈ {0.0, 0.1, 0.2, 0.3}`, fix others at default.
- **Compounded:** e.g. `(sequence_length=512, input_noise_rate=0.2)`.

Config files in `configs/` can instantiate these sweeps for data generation and evaluation scripts.
