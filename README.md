# Explainable Edge AI for Cost-Efficient and Trustworthy Anti-Money Laundering Systems

This repository contains the full implementation and experimental results for the paper **"Explainable Edge AI for Cost-Efficient and Trustworthy Anti-Money Laundering Systems"**.

The project develops a lightweight Anti-Money Laundering (AML) detection system under the framework **XE-AML**, combining ensemble-based machine learning, explainable AI, and Edge AI deployment principles. The central premise is that moving transaction screening closer to the transaction source — rather than routing every transaction to a centralized server — reduces detection latency, lowers infrastructure cost, and enables intervention before suspicious funds propagate further through the financial system.

Three ensemble models are evaluated: Random Forest, XGBoost, and LightGBM. SHAP is applied to the best-performing model to provide transparent, auditable explanations for each prediction. In addition to classification quality, the system is evaluated on computational efficiency metrics required for edge deployment: inference latency, model size, memory footprint, CPU utilization, and transaction throughput.

---

## Why This Project Exists

Traditional AML systems process transactions centrally after they have already been routed through the network. This introduces two compounding problems: detection latency allows funds to move before a suspicious flag is raised, and centralized processing concentrates infrastructure cost and creates a single point of failure.

XE-AML addresses both problems by positioning lightweight ML inference at the edge — at or near the point of transaction origination. The result is a system that can screen transactions in under 2 ms per transaction, operates on a model smaller than 1.1 MB, and requires less than 3 MB of additional memory at load time. This makes it deployable on commodity edge hardware without specialized accelerators.

Explainability is a second core requirement. Financial regulators increasingly require that automated AML decisions be interpretable by compliance officers, not just accurate. SHAP values provide per-transaction attribution — identifying precisely which features drove a suspicious classification — making the system auditable at the individual transaction level.

---

## Proposed System: XE-AML

XE-AML consists of three integrated layers:

**1. Transaction Detection Layer**
Intercepts incoming transactions and routes their feature vectors through the inference pipeline. Operates in near-real-time, designed to flag suspicious activity before settlement completes.

**2. Lightweight Machine Learning Layer**
LightGBM is selected as the production edge model. The rationale is threefold: it achieves the highest PR-AUC and ROC-AUC among the three candidates; it serializes to a 1.015 MB file, well within edge storage constraints; and its histogram-based tree algorithm is significantly faster at inference than Random Forest's averaging across deep trees or XGBoost's sequential boosting under high-load conditions. Random Forest was retained as a baseline but is unsuitable for edge deployment due to its larger memory footprint and lower detection quality on the dataset.

**3. Explainable AI Layer**
SHAP (SHapley Additive exPlanations) produces both global feature importance rankings and per-transaction local explanations. Global explanations support model auditing and regulatory documentation. Local waterfall plots allow a compliance officer to review exactly why a specific transaction was flagged — which features contributed positively toward a suspicious classification, and by how much. This satisfies the interpretability requirements outlined in financial compliance frameworks such as FATF recommendations and Basel AML guidelines.

---

## Dataset

**Dataset:** SAML-D (Synthetic Anti-Money Laundering Dataset)

**Target variable:**
```
Is_laundering
  0 = Normal transaction
  1 = Suspicious (money laundering) transaction
```

The dataset is severely class-imbalanced. Suspicious transactions constitute a small fraction of total transactions — a distribution that mirrors real-world AML data. This imbalance makes accuracy a misleading metric: a model that classifies every transaction as normal would score near-perfect accuracy while detecting nothing. PR-AUC (Precision-Recall Area Under Curve) is therefore the primary evaluation metric, as it directly measures performance on the minority class without being inflated by the majority.

| Aspect | Detail |
|---|---|
| Dataset name | SAML-D |
| Target variable | `Is_laundering` |
| Leakage columns removed | `Laundering_type`, `Type`, `Is_Suspicious` |
| Imbalance strategy | Internal model weighting — no oversampling |

**Why leakage columns are dropped:** `Laundering_type` encodes the category of money laundering, which is derived from the target label itself. `Is_Suspicious` is a near-duplicate of the target. `Type` carries transaction-type information that, in a real deployment, would not be available at inference time in the same form. Including any of these would allow the model to learn the label rather than the underlying transaction pattern, producing artificially inflated metrics that do not generalize.

Place `SAML-D.csv` in the project directory before running the notebook.

---

## Machine Learning Pipeline

### Step-by-Step

**Step 1 — Data loading and EDA**
Load the SAML-D dataset and inspect class distribution, missing values, and feature cardinalities.

**Step 2 — Leakage column removal**
Drop `Laundering_type`, `Type`, and `Is_Suspicious` before any feature computation. These columns are dropped first — before the split and before any feature engineering — to ensure that no downstream computation inadvertently inherits leakage information.

**Step 3 — Stateless feature engineering (pre-split)**
The following features are derived purely from values within each individual row, with no reference to the distribution of the dataset. They are safe to compute before the train/test split because no information from other rows is required:

| Feature | Construction | AML Rationale |
|---|---|---|
| `Amount_log` | `log1p(Amount)` | Transaction amounts span several orders of magnitude. Log-transform compresses extreme outliers while preserving the signal of unusually large transfers. |
| `CurrencyMismatch` | `Payment_currency ≠ Received_currency` | Currency conversion is a common layering technique in AML. Mismatched currencies flag potential cross-currency obfuscation. |
| `CrossBorder` | `Sender_bank_location ≠ Receiver_bank_location` | Cross-border transfers are higher-risk and subject to separate regulatory scrutiny. |
| `Hour`, `DayOfWeek`, `Month`, `IsWeekend` | Extracted from transaction timestamp | Suspicious transactions often cluster in off-hours or weekends when compliance monitoring is reduced. |

**Step 4 — Stratified train / validation / test split (70 / 15 / 15)**
The dataset is split into three non-overlapping subsets with stratification on the target label. The 70/15/15 ratio is chosen deliberately:

- The training set at 70% provides sufficient samples of the minority class for the model to learn meaningful decision boundaries despite the severe imbalance.
- A dedicated validation set (15%) is required to tune the classification threshold independently from the test set. Without a validation set, threshold optimization on the test set would constitute an evaluation leak — the test performance would reflect a threshold chosen to perform well on that specific subset rather than on unseen data.
- The test set (15%) is held out entirely and used only once, at the end, to report final metrics. It is never used to make any modeling decision.

**Step 5 — Post-split account behavioral feature engineering (train set only)**
Account-level behavioral features are computed after the split to prevent temporal and distributional leakage:

| Feature | Construction |
|---|---|
| `Sender_freq` | Proportion of transactions sent by this account (relative frequency) |
| `Receiver_freq` | Proportion of transactions received by this account |
| `Sender_txn_count` | Absolute count of transactions sent from this account |
| `Receiver_txn_count` | Absolute count of transactions received by this account |
| Account amount statistics | Per-account mean and maximum transaction amounts |

**Why these cannot be computed before the split:** These features aggregate information across multiple rows. Computing them on the full dataset before splitting would allow the validation and test sets to carry distributional information from training transactions — a form of target leakage. All aggregates are computed exclusively from the training partition and then mapped to validation and test rows using lookup joins.

`Sender_account` and `Receiver_account` are not encoded as raw categorical IDs. High-cardinality ID columns treated as categories cause models to memorize specific accounts rather than learn transferable AML patterns. The behavioral aggregates above capture the same information in a generalized, leak-safe form.

**Step 6 — Encoding and scaling (fit on train only)**
Categorical variables are encoded using `OrdinalEncoder`. Numerical features are standardized using `StandardScaler`. Both transformers are fit exclusively on the training set, then applied as read-only transforms to the validation and test sets. Fitting transformers on the full dataset before splitting is a common source of subtle data leakage and is avoided here.

**Step 7 — Model training with internal imbalance handling**
Three models are trained:

- **Random Forest** — `class_weight='balanced_subsample'` with `max_samples=0.2` for memory efficiency on large datasets.
- **XGBoost** — `scale_pos_weight = n_negative / n_positive`, which instructs the gradient boosting objective to assign proportionally higher loss to misclassified positive samples.
- **LightGBM** — `class_weight='balanced'`, which reweights the training samples inversely proportional to class frequency.

Oversampling techniques such as SMOTE or RandomOverSampler were evaluated but not used. Oversampling the training set increases the risk of overfitting to synthetic minority samples and, if applied before the split, leaks distributional information into the evaluation sets. Internal class weighting achieves equivalent imbalance correction without these risks.

**Step 8 — Threshold tuning on the validation set**
The default classification threshold of 0.5 is suboptimal for AML. The cost of a missed suspicious transaction (false negative) is substantially higher than the cost of a false alarm (false positive) in the context of financial crime prevention. Accordingly, the threshold is tuned to maximize **F2-Score** on the validation set. F2-Score applies a weight of 2 to recall relative to precision, reflecting the operational priority of minimizing missed detections over minimizing false alarms.

The tuned threshold is then applied to the test set. This separation — tune on validation, evaluate on test — ensures that reported test metrics reflect genuine generalization performance.

**Step 9 — SHAP explainability**

**Step 10 — Inference benchmark**

---

## Model Selection Rationale

### Why LightGBM

LightGBM is selected as the XE-AML production model based on two criteria evaluated jointly: detection quality and computational efficiency.

**Detection quality:** LightGBM achieves the highest PR-AUC (0.8666) and ROC-AUC (0.9976) among the three candidates on the test set. It also achieves the strongest F1-Score (0.8227) with a balanced precision-recall profile, which is important in an operational context where both false positives (wasted investigator time) and false negatives (missed crime) carry real costs.

**Computational efficiency:** LightGBM serializes to 1.015 MB on disk, loads with a memory footprint of only 2.85 MB, and achieves a mean single-transaction inference latency of 1.26 ms. These characteristics make it compatible with edge hardware. Random Forest, despite being a strong baseline, has substantially larger serialized size and memory requirements at equivalent tree depth. XGBoost achieves competitive detection metrics but offers no computational advantage over LightGBM in this configuration.

The combination of best-in-class detection and best-in-class computational efficiency makes LightGBM the unambiguous choice for an edge deployment scenario.

### Why PR-AUC over Accuracy

Accuracy rewards a model for correctly classifying the dominant class — normal transactions — regardless of whether it detects any suspicious transactions at all. On a severely imbalanced dataset, a trivial classifier that always predicts "normal" achieves near-perfect accuracy. PR-AUC measures the trade-off between precision and recall across all possible thresholds, focusing entirely on the minority class. It is the correct metric when the minority class is both rare and critical.

ROC-AUC is reported as a complementary metric. It is less sensitive to class imbalance than PR-AUC but provides a useful secondary signal about the model's overall discriminative ability.

---

## Classification Results

All metrics are evaluated on the held-out test set using the threshold tuned on the validation set.

| Model | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Tuned Threshold |
|---|---|---|---|---|---|---|
| Random Forest | 0.4598 | 0.6293 | 0.5314 | 0.5985 | 0.9924 | 0.8556 |
| XGBoost | 0.7843 | 0.8251 | 0.8042 | 0.8608 | 0.9970 | 0.9700 |
| **LightGBM** | **0.8244** | **0.8211** | **0.8227** | **0.8666** | **0.9976** | **0.9771** |

### Visualizations

#### Class Distribution
![Class Distribution](pencucian%20uang/pencucian%20uang/class_distribution.png)

#### Amount Distribution
![Amount Distribution](pencucian%20uang/pencucian%20uang/amount_distribution.png)

#### Confusion Matrices
![Confusion Matrices](pencucian%20uang/pencucian%20uang/confusion_matrices.png)

#### Precision-Recall Curve
![Precision-Recall Curve](pencucian%20uang/pencucian%20uang/precision_recall_curve.png)

#### ROC Curve
![ROC Curve](pencucian%20uang/pencucian%20uang/roc_curve.png)

#### Feature Importance (LightGBM)
![Feature Importance](pencucian%20uang/pencucian%20uang/feature_importance_best_model.png)

---

## Threshold Sweep — All Models

To support operational decision-making, each model was evaluated across five classification thresholds (0.5, 0.6, 0.7, 0.8, 0.9) on the validation set. Metrics reported are Precision, Recall, F1-Score, F2-Score, and the number of transactions predicted as suspicious (`predicted_suspicious`). The F2-Score is the primary sweep criterion, reflecting the operational priority of minimizing missed detections (false negatives carry higher cost than false positives in AML contexts).

| # | Threshold | Precision | Recall | F1 | F2 | Predicted Suspicious | Model |
|---|-----------|-----------|--------|----|----|----------------------|-------|
| 1 | 0.5 | 0.0051 | 0.8832 | 0.0096 | 0.2057 | 25,870 | Random Forest |
| 2 | 0.6 | 0.0087 | 0.8508 | 0.0148 | 0.2932 | 15,562 | Random Forest |
| 3 | 0.7 | 0.0143 | 0.8015 | 0.0243 | 0.4168 | 8,316 | Random Forest |
| 4 | 0.8 | 0.0287 | 0.7056 | 0.0408 | 0.5466 | 3,635 | Random Forest |
| 5 | 0.9 | 0.0660 | 0.5145 | 0.0581 | 0.5390 | 1,144 | Random Forest |
| 6 | 0.5 | 0.0060 | 0.9494 | 0.0148 | 0.3001 | 17,505 | XGBoost |
| 7 | 0.6 | 0.0105 | 0.9372 | 0.0188 | 0.3619 | 13,250 | XGBoost |
| 8 | 0.7 | 0.0143 | 0.9292 | 0.0248 | 0.4425 | 9,625 | XGBoost |
| 9 | 0.8 | 0.0217 | 0.9156 | 0.0352 | 0.5574 | 6,239 | XGBoost |
| 10 | 0.9 | 0.0392 | 0.8872 | 0.0544 | 0.7084 | 3,350 | XGBoost |
| 11 | 0.5 | 0.0077 | 0.9554 | 0.0135 | 0.2786 | 19,465 | LightGBM |
| 12 | 0.6 | 0.0092 | 0.9443 | 0.0167 | 0.3309 | 15,229 | LightGBM |
| 13 | 0.7 | 0.0120 | 0.9345 | 0.0213 | 0.3967 | 11,521 | LightGBM |
| 14 | 0.8 | 0.0170 | 0.9217 | 0.0288 | 0.4899 | 8,009 | LightGBM |
| 15 | 0.9 | 0.0300 | 0.9014 | 0.0450 | 0.6434 | 4,451 | LightGBM |

> **Note:** These results are evaluated on the validation set prior to final threshold tuning (Step 8). Precision values are low across all models because the dataset is severely class-imbalanced; the sweep is used to characterize recall-precision trade-offs across the threshold range, not as a standalone performance report. The tuned thresholds applied to the held-out test set are: **Random Forest → 0.8556**, **XGBoost → 0.9700**, **LightGBM → 0.9771** (see [Classification Results](#classification-results)).

### Key Observations

- **Random Forest** shows a steep precision-recall trade-off across thresholds. At threshold 0.5, recall reaches 0.8832 but precision collapses to 0.0051, indicating the model over-predicts suspicious transactions on this dataset. Even at threshold 0.9, precision (0.0660) remains far below XGBoost and LightGBM at equivalent recall levels — consistent with Random Forest's lower PR-AUC (0.5985).

- **XGBoost** maintains high recall (>0.88) across all thresholds while precision improves more sharply as the threshold rises. The F2-Score peaks at threshold 0.9 (0.7084), the highest single F2-Score in the sweep across all models at that threshold level.

- **LightGBM** shows the most stable recall across thresholds (0.9014–0.9554), but precision improvement with increasing threshold is more gradual than XGBoost. The F2-Score at threshold 0.9 (0.6434) is lower than XGBoost's (0.7084) at the same threshold, but LightGBM's advantage in PR-AUC and post-tuning F1 performance is preserved on the full threshold optimization curve. The validation-tuned threshold of 0.9771 reflects this: LightGBM requires a higher confidence score to achieve its optimal precision-recall balance.

- **F2-Score as the sweep criterion:** Because missed suspicious transactions carry higher operational cost than false alarms, F2-Score (which weights recall at 2× precision) is the appropriate metric for threshold selection. Sorting by F2-Score, the optimal pre-tuning threshold across the sweep is XGBoost at 0.9 (F2 = 0.7084), but LightGBM's full validation-set tuning procedure identifies a near-unity threshold (0.9771) that ultimately achieves a superior F1 (0.8227) and PR-AUC (0.8666) on the held-out test set.

---

## SHAP Explainability

Predictive accuracy alone is insufficient for production AML deployment. Regulators, compliance officers, and internal audit teams require that automated suspicious transaction classifications be explainable at the individual transaction level — not merely accurate in aggregate. This is reinforced by frameworks such as FATF Recommendation 16 and the EU's AMLD requirements, which emphasize that AML controls must be documented and defensible.

SHAP (SHapley Additive exPlanations) is applied to the LightGBM model to provide two levels of explanation:

**Global explanations** — feature importance across all test-set predictions, identifying which input features most consistently drive suspicious classifications across the portfolio. This supports model auditing, regulatory documentation, and ongoing model monitoring.

**Local explanations** — per-transaction waterfall plots that decompose the model's output probability into individual feature contributions for a single transaction. A compliance officer can review exactly which signals — an unusually large amount, a currency mismatch, an off-hours timestamp — pushed a transaction above the suspicious threshold, and by how much. This makes the classification defensible and reviewable without requiring the reviewer to understand the underlying model architecture.

Key features identified as most influential by SHAP: transaction amount, sender transaction count, sender amount statistics, currency mismatch, and payment type.

### SHAP Summary Plot
![SHAP Summary Plot](pencucian%20uang/pencucian%20uang/shap_summary_plot.png)

### SHAP Feature Importance
![SHAP Bar Plot](pencucian%20uang/pencucian%20uang/shap_bar_plot.png)

### SHAP Dependence Plot
![SHAP Dependence Plot](pencucian%20uang/pencucian%20uang/shap_dependence_plot.png)

### SHAP Waterfall — Suspicious Transaction
![SHAP Waterfall Suspicious](pencucian%20uang/pencucian%20uang/shap_waterfall_suspicious.png)

### SHAP Waterfall — Normal Transaction
![SHAP Waterfall Normal](pencucian%20uang/pencucian%20uang/shap_waterfall_normal.png)

---

## Inference Benchmark and Edge AI Suitability

The benchmark evaluates whether the LightGBM model meets the operational requirements for edge deployment. Six thresholds are assessed:

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Model size | < 10 MB | 1.015 MB | ✅ Pass |
| Memory footprint | < 50 MB | 2.85 MB | ✅ Pass |
| Mean latency | < 5 ms | 1.2647 ms | ✅ Pass |
| P99 latency | < 10 ms | 2.2710 ms | ✅ Pass |
| Throughput | > 1,000 TPS | 169,072 TPS | ✅ Pass |
| CPU per-core | < 80% | 72.90% | ✅ Pass |

### Why These Thresholds

**Model size < 10 MB:** Edge nodes commonly operate with storage constraints and limited over-the-air update bandwidth. A model under 10 MB can be deployed and updated on commodity hardware including embedded systems and regional gateway servers.

**Memory footprint < 50 MB:** Edge devices may run multiple concurrent processes. A model that requires hundreds of megabytes of RAM at load time is unsuitable for shared-resource environments.

**Mean latency < 5 ms / P99 < 10 ms:** Payment networks such as Visa and SWIFT process transactions with end-to-end routing windows measured in seconds. An AML screening step must complete well within that window to avoid introducing perceptible delays. Mean latency captures typical performance; P99 latency captures the worst-case behavior that determines whether SLAs can be guaranteed under production load. P99 is the operationally relevant figure for SLA design.

**Throughput > 1,000 TPS:** Even modest regional edge nodes must sustain throughput significantly above the peak transaction rate of the networks they serve. The 169,072 TPS batch throughput and 247,767 TPS stress-test throughput demonstrate substantial headroom above this floor.

**CPU per-core < 80%:** CPU utilization normalized per logical core must remain below 80% to preserve headroom for the operating system, network I/O, and co-located processes. The 72.90% figure falls within this bound.

### Benchmark Environment

| Parameter | Value |
|---|---|
| Platform | Windows |
| CPU logical cores | 8 |
| CPU physical cores | 6 |
| RAM total | 7.71 GB |
| Python version | 3.12.9 |
| Warm-up inferences | 200 (to eliminate JIT cold-start) |
| Latency measurement samples | 1,000 single-row inferences |

### Full Benchmark Results

| Metric | Value |
|---|---:|
| Mean latency | 1.2647 ms |
| Median latency | 1.1640 ms |
| P95 latency | 1.7895 ms |
| P99 latency | 2.2710 ms |
| Min / Max latency | 0.9042 / 2.8022 ms |
| Latency std dev | 0.2553 ms |
| Model size | 1.015 MB |
| Memory footprint (load) | 2.85 MB |
| Inference memory delta | 4.70 MB |
| Batch throughput (5,000 samples) | 169,072 TPS |
| Stress test throughput (≥5 s loop) | 247,767 TPS |
| Avg CPU per-core | 72.90% |
| CPU monitor samples valid | 48 / 48 |

### Measurement Methodology

**Memory footprint** is measured as the RSS (Resident Set Size) delta immediately before and after loading the model and scaler into memory. This isolates the memory cost attributable specifically to the model artifacts, excluding baseline process memory and Python runtime overhead.

**Inference memory delta** is measured as the additional RSS consumed during an active sustained inference loop, capturing allocations from repeated prediction calls beyond the initial load cost.

**CPU usage** is collected via a background thread polling `psutil.cpu_percent()` at 100 ms intervals during a sustained inference loop of at least 5 seconds. On Windows, `psutil` returns cumulative CPU across all logical cores (e.g., 8 cores at ~79% each = 583% total). This is divided by the number of logical cores to yield a per-core normalized figure (583% ÷ 8 = 72.90%). The 200-inference warm-up phase runs before measurement begins to eliminate first-call JIT overhead from the latency and CPU distributions.

**P95 and P99 latency** are the primary SLA-relevant figures. Mean latency can be misleadingly optimistic when the latency distribution has a right tail; P99 captures the worst-case single-row inference time that 99% of all transactions will complete within.

### Raw Benchmark Screenshots

#### Model Size and Memory Footprint
![Model Size and Memory](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_model_size_memory.png)

#### Inference Latency Distribution
![Inference Latency](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_latency.png)

#### Batch Throughput
![Batch Throughput](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_batch_throughput.png)

#### Stress Test — CPU and Memory
![Stress Test](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_stress_cpu_memory.png)

#### Edge AI Suitability Assessment
![Suitability Assessment](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_edge_ai_suitability.png)

---

## Repository Contents

```
README.md
.gitignore

pencucian uang/pencucian uang/
├── aml_detection_AML_FEATURE_ENGINEERING_FINAL.ipynb  ← main notebook (Sections 0–10)
├── SAML-D.csv                                          ← dataset (place here before running)
│
├── Model artifacts
│   ├── aml_best_model.pkl          ← serialized LightGBM model
│   ├── aml_scaler.pkl              ← fitted StandardScaler
│   ├── aml_encoder.pkl             ← fitted OrdinalEncoder
│   ├── aml_threshold.pkl           ← optimal classification threshold (from validation set)
│   ├── aml_best_scores.pkl         ← best model evaluation scores dict
│   ├── aml_account_stats.pkl       ← account behavioral statistics (computed from train set)
│   ├── aml_feature_columns.pkl     ← ordered list of final input feature columns
│   ├── aml_numeric_medians.pkl     ← per-column medians for missing value imputation
│   ├── aml_cat_cols.pkl            ← list of categorical columns
│   ├── tmp_LightGBM.pkl            ← intermediate LightGBM checkpoint
│   ├── tmp_XGBoost.pkl             ← intermediate XGBoost checkpoint
│   └── tmp_Random_Forest.pkl       ← intermediate Random Forest checkpoint
│
├── Evaluation outputs
│   ├── aml_model_comparison_results.csv    ← per-model metric summary
│   ├── threshold_report_all_models.csv     ← threshold sweep results across all models
│   ├── threshold_report_lightgbm.csv
│   ├── threshold_report_xgboost.csv
│   └── threshold_report_random_forest.csv
│
├── Benchmark outputs
│   ├── inference_metrics_updated.json      ← latest benchmark results (JSON)
│   ├── inference_metrics_refactored.json
│   └── inference_metrics.json
│
├── Figures
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── confusion_matrices.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   ├── feature_importance_best_model.png
│   ├── shap_summary_plot.png
│   ├── shap_bar_plot.png
│   ├── shap_dependence_plot.png
│   ├── shap_waterfall_suspicious.png
│   └── shap_waterfall_normal.png
│
└── benchmark_outputs/
    ├── benchmark_model_size_memory.png
    ├── benchmark_latency.png
    ├── benchmark_batch_throughput.png
    ├── benchmark_metrics_json.png
    ├── benchmark_stress_cpu_memory.png
    └── benchmark_edge_ai_suitability.png
```

---

## Paper Context

> **"Explainable Edge AI for Cost-Efficient and Trustworthy Anti-Money Laundering Systems"**
> Bella Adisty, Nela Eka Silvia Lumbanraja, Albert Jofrandi Hutapea
> Program Studi Informatika, Institut Teknologi Sains Bandung
> Innovation Paper Competition 2026 — Risk and Governance Summit

This implementation operationalizes four claims made in the paper:

1. A LightGBM model trained on engineered AML features achieves PR-AUC of 0.8666 — substantially above the Random Forest baseline (0.5985) on the same dataset and split.
2. The model serializes to under 2 MB combined (model + scaler) and loads with a memory footprint of 2.85 MB, satisfying edge hardware constraints.
3. Single-transaction inference completes in 1.26 ms on average, with P99 under 2.3 ms — sufficient for real-time screening within standard payment network latency windows.
4. SHAP explanations provide per-transaction attribution that satisfies interpretability requirements for compliance and regulatory review.

---

## Reproducibility Notes

- Run all cells in order from Section 0 (library installation) through Section 10 (inference benchmark).
- The notebook may not render fully on GitHub when cell outputs are large. Figures are provided as separate PNG files for reference.
- CPU usage is reported as a per-core normalized value. On multi-core Windows systems, raw `psutil.cpu_percent()` values exceeding 100% reflect cumulative multi-core usage and must be divided by logical core count to obtain per-core utilization.
- All preprocessing artifacts (scaler, encoder, account stats, feature column list, threshold) must be saved and loaded together during inference. The notebook's Section 9 demonstrates the complete inference pipeline including all preprocessing steps.
