# SafeShift

Benchmarking test-time adaptation methods on CIFAR-100-C to see how well models can cope with distribution shift at inference time, without retraining.

The idea is simple: a model trained on clean data will often fall apart when it sees corrupted or shifted inputs in the real world. Test-time adaptation (TTA) tries to fix that by adapting the model on the fly using only the incoming test data. This project implements and compares a few popular TTA methods to see what actually works, what breaks, and under what conditions.

## Methods

| Method | Paper | Status |
|--------|-------|--------|
| Source (no adaptation) |  | In progress |
| [Tent](https://arxiv.org/abs/2006.10726) | Wang et al., 2021 | In progress |
| [EATA](https://arxiv.org/abs/2204.02610) | Niu et al., 2022 | In progress |

**Source** is the baseline, just evaluate the pretrained model on corrupted data with no adaptation at all. **Tent** minimises the entropy of model predictions by updating the batch norm parameters at test time. **EATA** builds on that with sample-aware entropy filtering and anti-forgetting regularisation to keep things more stable.

## Dataset

Uses [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) for training and [CIFAR-100-C](https://zenodo.org/records/3555552) for evaluation under distribution shift. CIFAR-100-C contains 19 different corruption types (gaussian noise, blur, weather, digital artifacts, etc.) at 5 severity levels each.

## Project Structure

```
safeshift/
├── src/
│   ├── data.py           # data loading for CIFAR-100 and CIFAR-100-C
│   ├── models.py         # model definitions
│   ├── adaptation.py     # TTA method implementations
│   ├── evaluation.py     # evaluation loops and metrics
│   └── utils.py          # config loading, logging, misc helpers
├── scripts/
│   ├── train_source.py       # train source model on clean CIFAR-100
│   ├── run_experiment.py     # run TTA evaluation on corrupted data
│   └── aggregate_results.py  # collect and summarise results
├── configs/
│   ├── source.yaml       # source model training config
│   ├── tent.yaml         # Tent hyperparameters
│   └── eata.yaml         # EATA hyperparameters
├── notebooks/            # exploratory analysis and pytorch practice
├── tests/                # unit tests
├── data/                 # datasets (not tracked)
├── checkpoints/          # saved model weights (not tracked)
├── outputs/              # experiment results (not tracked)
└── experiments/          # experiment logs
```

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/emmaw04/safeshift.git
cd safeshift
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pandas pillow
```

You'll also need to download the datasets. CIFAR-100 downloads automatically through torchvision, but CIFAR-100-C needs to be grabbed manually:

```bash
mkdir -p data
# CIFAR-100-C
wget -P data/ https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
tar -xf data/CIFAR-100-C.tar -C data/
```

## Usage

**Train the source model:**
```bash
python scripts/train_source.py --config configs/source.yaml
```

**Run TTA evaluation:**
```bash
python scripts/run_experiment.py --config configs/tent.yaml
```

**Aggregate results:**
```bash
python scripts/aggregate_results.py
```

## Current Status

**Work in progress.** The data pipeline for CIFAR-100 and CIFAR-100-C is done. Currently working on the source model training loop and adaptation method implementations. Configs and evaluation scripts are still being fleshed out.

## References

- Wang et al. [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726) (ICLR 2021)
- Niu et al. [Efficient Test-Time Model Adaptation without Forgetting](https://arxiv.org/abs/2204.02610) (ICML 2022)
- Hendrycks & Dietterich [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261) (ICLR 2019)
