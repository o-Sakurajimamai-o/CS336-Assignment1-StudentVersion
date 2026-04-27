# CS336 Assignment 1 Student Version

Repository: <https://github.com/o-Sakurajimamai-o/CS336-Assignment1-StudentVersion.git>

This repository contains my student implementation work for Stanford CS336
Assignment 1. The project focuses on building a BPE tokenizer and core deep
learning modules from scratch.

This is a student version of the assignment code. I implemented the core pieces
myself while following the assignment specification, with the goal of
understanding how tokenization and the earliest neural network layers work under
the hood. The code is intentionally kept readable and direct, so it can show the
learning process rather than hide everything behind a polished framework.

## Scope

This repository is meant to grow with the assignment. It will contain my own
implementations of the components required by CS336 Assignment 1, such as
tokenization, model layers, optimization utilities, and training helpers.

## Project Layout

```text
cs336_basics/
  classes.py
  function.py
```

## Dependencies

The project uses Python 3.12+ and the following main packages:

- `regex`
- `torch`
- `tqdm`

Install dependencies with:

```bash
pip install -e .
```

## Notes

The BPE trainer starts from the 256 raw byte tokens, adds configured special
tokens, counts adjacent byte-pair frequencies, and repeatedly merges the most
frequent pair. The tokenizer applies learned merges by rank and preserves
special tokens as indivisible tokens.

Large datasets, generated vocabulary files, assignment tests, local virtual
environment files, and course handouts are intentionally not included. This
repository keeps only the student implementation code needed to show the core
work.
