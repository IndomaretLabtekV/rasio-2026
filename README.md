# HARVEST: Food System Resilience Prediction

**H**istorical **A**nd **R**egional **V**ariables for **E**stimating **S**ystem resilience **T**rends. Final submission for the [DSC RASIO 10.0 (2026) Kaggle competition](https://www.kaggle.com/competitions/food-system-resilience-prediction-DSC-RASIO-2026/).

The task is to forecast the 2026 `food_system_resilience_index` for 358 regions from multi-domain annual predictors (agriculture, climate, economy, logistics, social nutrition, shock history) and a regional neighbor graph. Metric: RMSE.

## Results

| Leaderboard | RMSE |
|---|---|
| Public | **8.4559** |
| Private | **8.4261** |

Validation (nested rolling-origin, 2018–2025): pooled OOF RMSE **9.349** (18% below the persistence baseline), 2024–2025 RMSE **9.243**.

## Approach

- **Key EDA findings.** Resilience is region-structured (~50% of variance comes from region identity), persistent (AR(1) φ = 0.75) and spatially correlated (graph homophily 0.87). The strongest same-year predictor, the post-harvest assessment (r = 0.80), is 100% missing for 2026.
- **Leakage-safe features.** 61 causal features (the F_N set): lags, rolling means and historical means of the target, plus `nb_thm`, the weighted mean of neighbors' historical target means. Every history uses years ≤ t−1 only. Same-year post-harvest and target values are never used.
- **Model.** One untuned CatBoost (CPU, ordered boosting) averaged over five seeds. No pretrained weights, no external data.
- **Validation.** Each year 2018–2025 is scored by a model trained only on earlier years. Tree count is chosen on year t−1, never on the scored year. Models are selected on historical validation only, never on the Public LB.
- **Rejected challenger.** A forward-trained post-harvest proxy (+1 feature) won only 2 of 8 years and lost 2024–2025, so it was not used.

### Benchmark

| Model | Features | OOF RMSE | 2024–25 RMSE |
|---|---|---|---|
| Persistence | target lag-1 | 11.370 | 11.816 |
| LightGBM | F_N (61) | 9.556 | 9.455 |
| XGBoost | F_N (61) | 9.541 | 9.261 |
| CatBoost | F_0 (60) | 9.446 | 9.380 |
| CatBoost | F_H (18) | 9.333 | 9.277 |
| ExtraTrees | F_N (61) | 9.288 | 9.302 |
| Hybrid ensemble | CB + EB + ET | **9.274** | 9.279 |
| CatBoost + PH proxy | F_N + proxy (62) | 9.363 | 9.253 |
| **HARVEST (CatBoost)** | **F_N (61)** | 9.349 | **9.243** |

HARVEST has the lowest RMSE on 2024–2025, the folds closest to the 2026 setting, so it was selected over the hybrid ensemble.

## Repository

```
.
├── IndomaretLabtekV/
│   ├── notebook1.ipynb   # v24
│   ├── notebook2.ipynb   # v25
│   ├── notebook3.ipynb   # v26
│   └── submission{1,2,3}.csv
└── 007DSC26_Presentation File.pdf
```

Each notebook covers the full pipeline: data assembly, EDA, feature engineering, walk-forward validation, SHAP explainability, 2026 inference and submission.

## Reproduce

The notebooks are built for Kaggle and read data from the mounted competition input.

1. Open a notebook on Kaggle and add the competition dataset `food-system-resilience-prediction-DSC-RASIO-2026`.
2. Run all cells. Dependencies are pinned in the first code cell (`catboost==1.2.10`, `pandas==2.3.3`, `scikit-learn==1.6.1`, ...). A GPU is not required.

## Team DoaAyahIbu

- Riantama Putra
- Athilla Zaidan Zidna Fann
- Nicholas Wise Saragih Sumbayak

Full method and analysis: [presentation slides](007DSC26_Presentation%20File.pdf).
