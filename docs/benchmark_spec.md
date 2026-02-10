# Benchmark Specification

## Scope and Assumptions

SeqFactorBench evaluates sequence models under **controlled, compounded difficulty**. The benchmark isolates and combines four orthogonal factors to expose failure modes in accuracy, stability, and efficiency that do not appear under single-factor evaluation.

**Assumptions:**

- Tasks are **synthetic** with fully specified ground truth (no ambiguity).
- Models are evaluated on **sequence-to-label** and **sequence-to-sequence** prediction.
- Evaluation is **model-agnostic** (recurrent, attention-based, state-space).
- Factors are **orthogonal** and can be swept independently or in combination.

---

## Factor Definitions and Ranges

### 1. Scale

**Definition:** Length and size of sequences and vocabularies.

| Sub-factor      | Description                    | Range (min → max) | Unit / Notes        |
|-----------------|--------------------------------|-------------------|---------------------|
| Sequence length | Number of tokens per input     | 16 → 4096         | tokens              |
| Vocabulary size | Cardinality of input symbols   | 32 → 65536        | symbols             |
| Batch size      | Number of sequences per batch  | 1 → 256           | sequences           |

**Rationale:** Stress-tests memory, long-range dependencies, and throughput.

---

### 2. Breadth

**Definition:** Diversity of task types and number of distinct operations or labels.

| Sub-factor       | Description                         | Range (min → max) | Unit / Notes     |
|------------------|-------------------------------------|-------------------|------------------|
| Task family count| Number of synthetic task families   | 1 → 16            | families         |
| Label / output size | Cardinality of outputs per task | 2 → 1024          | classes / symbols|
| Num. operations  | Distinct operations in a task mix   | 1 → 32            | operations       |

**Rationale:** Measures generalization across task types and label spaces.

---

### 3. Structure

**Definition:** Recursion, nesting, and dependency patterns in the data.

| Sub-factor        | Description                          | Range (min → max) | Unit / Notes      |
|-------------------|--------------------------------------|-------------------|-------------------|
| Nesting depth     | Max depth of recursive structure      | 1 → 16            | depth             |
| Dependency span   | Max distance of a dependency         | 1 → 512           | tokens            |
| Grammar complexity| Number of rules / states in generator | 2 → 64            | rules / states    |

**Rationale:** Probes ability to handle hierarchical and long-range structure.

---

### 4. Noise

**Definition:** Corruption and stochasticity in inputs or labels.

| Sub-factor      | Description                          | Range (min → max) | Unit / Notes     |
|-----------------|--------------------------------------|-------------------|------------------|
| Input noise rate| Fraction of tokens randomly replaced | 0.0 → 0.5         | [0, 1]           |
| Label noise rate| Fraction of labels flipped           | 0.0 → 0.3         | [0, 1]           |
| Dropout (eval)  | Optional input dropout at eval       | 0.0 → 0.2         | [0, 1]           |

**Rationale:** Tests robustness to distribution shift and label noise.

---

## Compounding and Sweeps

- Each factor can be swept **independently** (other factors held at baseline).
- **Compounded** runs combine two or more factors (e.g., long sequences + high noise).
- Baseline defaults for sweeps will be defined in `parameters.md` and configs.

## Out of Scope

- Natural language understanding, world knowledge, or instruction following.
- Effects of pretraining data or domain shift from real-world corpora.
