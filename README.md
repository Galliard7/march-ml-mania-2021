# March Machine Learning Mania 2021

Predicting NCAA basketball tournament game outcomes for both men's (NCAAM) and women's (NCAAW) brackets. This was a [Kaggle competition](https://www.kaggle.com/competitions/ncaam-march-mania-2021) in 2021, scored on log loss. Notebooks developed on [Kaggle](https://www.kaggle.com/illidan7).

## Approach

### 1. NCAAM Feature Pipeline

Builds 30+ engineered features per team from historical game data: cumulative and season-level win ratios segmented by venue (home, away, neutral court), head-to-head records between matchup pairs, tournament seed rankings, and Massey ordinal system rankings. Pairwise matchup features include seed gap, win ratio ratio, and head-to-head dominance score.

### 2. NCAAM Final Submission

XGBoost classifier trained on the NCAAM feature matrix with GroupTimeSeriesSplit cross-validation that respects temporal ordering (no future leakage). Feature importance pruning removes low-signal columns. Outputs calibrated win probabilities for all possible tournament matchups.

### 3. NCAAW Feature Pipeline

Same feature engineering pipeline adapted for the women's tournament data. Handles differences in data availability (fewer seasons of historical data, different team pools) while maintaining the same 30+ feature structure for consistency.

### 4. NCAAW Final Submission

Separate XGBoost model trained on NCAAW features using the identical pipeline architecture. Women's tournament has different dynamics (higher seed predictability, fewer upsets) that the model captures through the same feature set with independently tuned parameters.

## Repository Structure

```
march-ml-mania-2021/
├── README.md
├── .gitignore
└── notebooks/
    ├── 01-ncaam-feature-pipeline.ipynb              # 30+ features: venue splits, seeds, ordinals
    ├── 02-ncaam-final-submission.ipynb              # XGBoost with GroupTimeSeriesSplit CV
    ├── 03-ncaaw-feature-pipeline.ipynb              # Same pipeline for women's tournament
    └── 04-ncaaw-final-submission.ipynb              # Separate NCAAW XGBoost model
```

## Tech Stack

- **ML**: XGBoost, scikit-learn
- **Data**: pandas, NumPy
- **Validation**: GroupTimeSeriesSplit (custom temporal CV)

## Competition

| | |
|---|---|
| **Competition** | [NCAA March Machine Learning Mania 2021](https://www.kaggle.com/competitions/ncaam-march-mania-2021) |
| **Type** | Featured prediction (tabular, probability calibration) |
| **Metric** | Log Loss |
| **Timeline** | February 2021 -- April 2021 |
