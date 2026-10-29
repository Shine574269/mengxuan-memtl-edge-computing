# Multi-Head Learning for Edge Computing Offloading

Final-year thesis code by **Mengxuan Shen** (BEng Digital Media Technology, Xi'an Jiaotong-Liverpool University, 2026).

Portfolio page: https://shine574269.github.io/projects/memtl-edge-computing.html · Thesis: [`thesis.pdf`](thesis.pdf) · 中文说明: [`README_zh.md`](README_zh.md)

## What it does

In multi-access edge computing, each mobile user decides whether to run a task locally or offload it to an edge server. Finding the optimal decision by exhaustive search grows exponentially with the number of users. This project trains neural networks to predict the decision directly from six features per user (input size, CPU cycles, local CPU frequency, channel gain, latency weight, energy weight).

- **MTL** – single-head multi-task feed-forward baseline.
- **MEMTL** – a shared backbone with several specialised prediction heads, trained in stages (backbone, then each head on its own data subset), with heads added dynamically.

## My contributions

This repository builds on the open-source [MTFNN-CO](https://github.com/qiyu3816/MTFNN-CO) implementation by its original authors (IEEE TMC 2020; IEEE GLOBECOM 2023). My changes:

| Change | Why |
|---|---|
| Rewrote `MEMTL.train_step` to compute the loss per head manually, with global-norm gradient clipping (1.0) and filtering of `None` gradients | The multi-head output list broke Keras' compiled loss ("No gradients" crash) and training was unstable |
| Unlock/lock the layer tracker when adding heads | Keras 3 otherwise blocks adding heads after the model is built |
| XLA-compiled inference (`tf.function(jit_compile=True)`) with warm-up, calling the model directly instead of `predict()` | Per-sample prediction latency fell to about 0.01–0.02 ms |
| Separator auto-detection in `utils/dataset.py` | The 6–8 user datasets are tab-separated |
| Keras 3 save/load API updates in `utils/model_op.py` | Compatibility with current Keras |
| Added 6/7/8-user configurations and `example/evaluate.py` (accuracy, precision, recall, F1) | Extended evaluation |

The label-free head-selection study (voting, input-only gating and enhanced gating) is described in the thesis.

## Results (five heads vs. MTL baseline)

| Users | Samples | MEMTL accuracy | MTL accuracy | MEMTL latency |
|---|---|---|---|---|
| 2 | 40K | **94.96%** | 93.88% | 0.0189 ms |
| 3 | 50K | **87.83%** | 87.25% | 0.0207 ms |
| 4 | 80K | **85.22%** | 80.34% | 0.0177 ms |
| 5 | 100K | **77.22%** | 74.94% | 0.0194 ms |
| 6 | 200K | **80.89%** | 78.65% | 0.0190 ms |
| 7 | 250K | **80.34%** | 77.98% | 0.0211 ms |
| 8 | 300K | **77.29%** | 74.45% | 0.0202 ms |

MEMTL accuracy here uses oracle head selection (an upper bound); see the thesis for deployable, label-free selection. Full logs: [`experiment.log`](experiment.log), [`results/`](results/).

## Repository layout

```
model/         MTL and MEMTL models
utils/         dataset loading/generation, cost calculation, loss, model save/load
example/       training scripts (MTL_example.py, MEMTL_example.py) and evaluate.py
config/        experiment configurations
ready_models/  trained MEMTL weights for 2–8 users and 2–5 heads
results/       per-dataset predictions and summary
datasets/      2-user sample dataset (others omitted for size)
```

## Running

```bash
pip install -r requirements.txt          # Python 3.10
python example/MEMTL_example.py          # full experiment (several hours)
python example/evaluate.py               # evaluate saved models
```

Only the 2-user dataset (5 MB) is included; the 3–8 user datasets (464 MB in total, some over GitHub's 100 MB file limit) are available on request; new datasets of the same form can be generated with `utils/dataset_generate.py` (`CO_MINLP_GEN`).
