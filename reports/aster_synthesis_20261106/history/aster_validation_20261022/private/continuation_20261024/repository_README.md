> Source snapshot: /home/gcx/aster_validation_20261022/private/continuation_20261024/repository_README.md
> Dates and execution states below refer to that original report. Relative artifact paths refer to the source run.

# MedGUIDE: A Benchmark for Clinical Guideline-Based Reasoning in LLMs

---

## Overview

**MedGUIDE** is a multiple-choice QA benchmark designed to evaluate whether language models can make clinically appropriate decisions by reasoning over structured medical guidelines (e.g., NCCN oncology protocols).

The benchmark consists of 8K synthetic clinical scenarios generated via decision trees from 17 cancer types. Each sample includes:
- A patient vignette
- A diagnostic or treatment decision point
- 4–6 answer options grounded in the guideline path

---

## Directory Structure

```
Submission-MedGUIDE/
├── data/
│   ├── MedGUIDE-8K.jsonl        # Main QA dataset
│   ├── all_paths.json           # Extracted decision paths
│   ├── decision_trees/          # Guideline-derived tree structures
│   └── guideline_docs/          # Source NCCN diagrams (PDF/images)
│
├── generate_profiles.py         # Profile question generation for probing models
├── generate_responses.py        # Run LLMs on the benchmark and save outputs
├── evaluate.py                  # Score model predictions vs. gold answers
│
├── utils.py                     # Shared utility functions
├── templates.py                 # Prompt templates for QA generation
├── README.md                    # This file
```

---

## Usage

### 1. Generate Model Responses

To run a closed-source LLM (e.g., GPT-4.1) on the MedGUIDE benchmark:

```bash
python generate_responses.py --model gpt-4.1
```

Responses will be saved to `benchmark_results/benchmark_gpt-4.1.jsonl`.

### 2. Evaluate Performance

Once responses are saved:

```bash
python evaluate.py --model gpt-4.1
```

This computes both:
- **Accuracy**: exact match rate
- **Weighted Accuracy**: downweights random guess chance (based on number of options)

---

## License

All data is synthetic and derived from publicly available NCCN guidelines. No patient information is included.
