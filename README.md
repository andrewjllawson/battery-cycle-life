# Battery cycle-life prediction from early cycling data

Predicting how many cycles a lithium-ion cell will last using only its first 100 cycles, with the open dataset from Severson et al. (2019).

> **Status:** week 1 of 8, data loading and exploration. Results table and key figure go here once modelling starts.

## Prior work

This project starts by replicating [Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 383–391 (2019)](https://www.nature.com/articles/s41560-019-0356-8). They predicted cycle life of 124 commercial LFP/graphite cells to 9.1% test error from the first 100 cycles. Their modelling code is under an academic licence, so all models here are written from scratch.

## Data

Download the three main batch files from [data.matr.io/1](https://data.matr.io/1/) into `data/raw/`:

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `2017-06-30_batchdata_updated_struct_errorcorrect.mat`
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat`

These are not the low-rate files used for the paper's Figure 4.

`src/load_data.py` converts them to pickles and applies the authors' cleaning from [their data-processing repo](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation). It removes cells that never reach 80% capacity, joins five batch-1 cells that continued in batch 2, and drops noisy batch-3 channels, giving 124 cells: 41 train, 43 test and 40 secondary test.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.load_data            # converts data/raw/*.mat -> data/processed/*.pkl
jupyter lab notebooks/01_explore.ipynb
```

## Repo layout

```
src/load_data.py      .mat -> pickle, cleaning, train/test split
src/features.py       ΔQ(V) and other features
notebooks/            exploration and modelling, numbered in order
results/figures/      saved plots
results/metrics.csv   one row per model per test set (from week 3)
```

## Citation

If you use the data, cite: Severson, K.A., Attia, P.M., et al. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* 4, 383–391 (2019).
