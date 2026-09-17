# Tensorflow-Audiobooks-Preprocessing

Udemy Course: Deep Learning Activity

# Audiobooks App — Customer Conversion Prediction

## Business Case

An audiobook app wants to predict whether a customer who has already made at least one purchase is **likely to buy again**. The goal is to target advertising spend at customers likely to convert, rather than wasting budget on customers unlikely to return — improving sales and profitability.

This is a **binary classification problem**: will the customer buy again, or not.

### Dataset Columns

| Column                       | Description                                                                                                                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ID                           | Customer identifier                                                                                                                                                                                  |
| Book length overall          | Sum of lengths of all purchases                                                                                                                                                                      |
| Book length avg              | Book length overall ÷ number of purchases                                                                                                                                                            |
| Price overall                | Sum of price of all purchases (strong predictor)                                                                                                                                                     |
| Price avg                    | Price overall ÷ number of purchases                                                                                                                                                                  |
| Review                       | Boolean — did the customer leave a review? (reviewers are more likely to convert)                                                                                                                    |
| Review 10/10                 | Average review score (1–10). Missing values are filled with the dataset average (8.91), which acts as a status-quo baseline — above it signals better-than-average sentiment, below it signals worse |
| Minutes listened             | Engagement measure                                                                                                                                                                                   |
| Completion                   | Minutes listened ÷ total length of purchased books                                                                                                                                                   |
| Support requests             | Number of support tickets (engagement/friction signal)                                                                                                                                               |
| Last visited − Purchase date | Engagement signal — a larger gap suggests the customer is still active and likely to return; 0 means the customer never returned (or only visited on the first day)                                  |
| Targets                      | Boolean — did the customer purchase again within the following 6 months?                                                                                                                             |

The dataset spans 2 years of customer engagement, with the target label based on purchase behavior in a subsequent 6-month window (supervised learning).

### Business Action Plan

1. Preprocess the data
2. Balance the dataset (equal priors between "bought again" and "did not")
3. Split into train / validation / test sets
4. Save in a tensor-friendly format (`.npz`)
5. Build and train the ML model

### Expected Accuracy Benchmarks

- **~70%** — acceptable
- **~80%** — good
- **~90%** — very good / impressive

---

## Model Experiments

All models use a `Dense` feedforward network with `relu` hidden layers, `softmax` output, `adam`-family optimizer, `sparse_categorical_crossentropy` loss, and early stopping on validation loss. Batch size 100 unless noted.

| Hidden size | Layers | Patience | Learning rate | Dropout | Test accuracy | Test loss |
| ----------- | ------ | -------- | ------------- | ------- | ------------- | --------- |
| 50          | 2      | 5        | default       | —       | **81.70%**    | 0.35      |
| 100         | 2      | 5        | default       | 0.2     | 80.80%        | 0.36      |
| 100         | 3      | 5        | 0.001         | —       | 80.13%        | 0.35      |
| 100         | 3      | 2        | 0.001         | —       | 78.57%        | 0.37      |
| 100         | 3      | 5        | 0.01          | —       | 79.24%        | 0.36      |
| 100         | 4      | 5        | 0.001         | —       | 81.03%        | 0.35      |
| 100         | 5      | 5        | 0.001         | —       | 79.91%        | 0.37      |

### Observations

- **Validation accuracy consistently plateaus in the high-70s to low-80s** across every configuration tried, regardless of depth, width, learning rate, or dropout.
- **The smallest model (2 layers, 50 units) performed best** (81.70% test accuracy), suggesting the extra capacity in larger networks isn't being converted into better generalization — likely because the bottleneck is the input features, not model capacity.
- **Increasing depth (3–5 layers) did not improve results** and in some cases hurt performance slightly, a sign of diminishing returns or mild overfitting relative to the amount of training data available.
- **Raising the learning rate to 0.01** made training noisier without any accuracy benefit.
- **Lower patience (2) hurt performance** — the model needs a few extra epochs of tolerance past the best validation loss to fully converge.
- **Dropout (0.2) did not help** — a sign that overfitting isn't the primary limiting factor here.

---

## Conclusion

Across every architecture and hyperparameter combination tested, test accuracy consistently lands in the **78–82% range**, with the simplest model (2 hidden layers, 50 units, patience 5) performing best at **81.70%**. This strongly suggests the model has reached a **practical performance ceiling set by the available features**, not by model capacity or tuning.

The 10 engineered features (book length, price, reviews, engagement, support requests) carry a meaningful but limited amount of signal about whether a customer will return — adding more layers, units, or dropout doesn't help because the model isn't underfitting or overfitting in a way more capacity or regularization can fix. It's a **data ceiling, not a modeling ceiling**.

**By the benchmarks in the business case, ~80% test accuracy is a "good" result** and a reasonable stopping point for this dataset. To meaningfully push past ~85–90%, the next step would likely need to come from **better or additional features** (e.g. more granular engagement data, genre preferences, time-series purchase patterns) rather than further architecture or hyperparameter tuning.

### Recommended next steps

- Treat the 2-layer, 50-unit, patience-5 configuration as the baseline/production model.
- Investigate feature engineering (interaction terms, time-based features) before investing further in architecture search.
- Consider simpler, faster-to-train alternatives (e.g. gradient boosted trees) as a sanity check — tree-based models often match or beat small neural nets on tabular data like this.
- If available, gather richer behavioral data to raise the ceiling on achievable accuracy.
