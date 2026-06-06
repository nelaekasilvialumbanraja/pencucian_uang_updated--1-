# 🔍 Explainable Edge AI for Cost-Efficient and Trustworthy Anti-Money Laundering Systems

This repository contains the implementation and experimental results for the paper **"Explainable Edge AI for Cost-Efficient and Trustworthy Anti-Money Laundering Systems"**.

The project develops a lightweight Anti-Money Laundering (AML) detection system using machine learning and explainable artificial intelligence under the framework named **XE-AML**. The main objective is to support cost-efficient and trustworthy AML detection by moving transaction screening closer to the transaction source through an Edge AI approach.

The proposed system evaluates three ensemble-based models — Random Forest, XGBoost, and LightGBM — to detect suspicious financial transactions. SHAP is used to explain model predictions and support transparency, auditability, and regulatory review.

In addition to classification performance, this project evaluates computational performance for inference deployment, including inference latency, model size, memory consumption, CPU usage, and transaction throughput.

---

## Project Overview

This project includes:

- Anti-Money Laundering transaction classification using ensemble-based machine learning
- Data preprocessing and feature engineering on the SAML-D dataset
- Leakage-safe feature engineering: stateless features computed before split; account behavioral features computed from the training set only
- Model training and evaluation: Random Forest, XGBoost, and LightGBM
- Class imbalance handling via `class_weight='balanced'` / `scale_pos_weight` — no oversampling applied
- Threshold tuning using F2-Score on the validation set; test set used only for final evaluation
- Model evaluation using Precision, Recall, F1-Score, ROC-AUC, and PR-AUC
- SHAP-based explainability for suspicious transaction interpretation
- Inference benchmark for Edge AI computational performance evaluation
- Measurement of model size, memory footprint, CPU usage, latency, and throughput

---

## Proposed System: XE-AML

XE-AML (Explainable Edge AI for Anti-Money Laundering) is designed to support near-real-time suspicious transaction detection by deploying a lightweight machine learning model closer to transaction origination points.

Instead of relying only on centralized data centers, the Edge AI approach allows transaction features to be processed at local or regional edge nodes. This reduces communication latency, lowers infrastructure cost, and supports faster detection before suspicious funds move further through the financial system.

The system consists of three main components:

1. **Transaction Detection Layer**
   Detects whether a transaction is normal or suspicious using lightweight machine learning.

2. **Lightweight Machine Learning Layer**
   Uses LightGBM as the main edge model because it provides strong predictive performance with relatively low computational cost, small model size, and efficient memory usage.

3. **Explainable AI Layer**
   Uses SHAP to explain why a transaction is classified as suspicious or normal, supporting compliance officer review and regulatory audit.

---

## Dataset

This project uses the **SAML-D dataset**, a transaction-level dataset for anti-money laundering detection.

The target variable is:

```
Is_laundering
  0 = Normal transaction
  1 = Suspicious transaction
```

The dataset is highly imbalanced. Class imbalance is handled internally by each model via `class_weight='balanced'` (Random Forest, LightGBM) and `scale_pos_weight` (XGBoost). No subsampling or oversampling is applied to preserve data integrity and avoid methodological leakage.

| Dataset Aspect | Description |
|---|---|
| Dataset name | SAML-D |
| Target variable | `Is_laundering` |
| Dropped leakage columns | `Laundering_type`, `Type`, `Is_Suspicious` |

Place `SAML-D.csv` in the project directory before running the notebook.

---

## Machine Learning Pipeline

The experimental pipeline follows these steps:

1. Load SAML-D dataset
2. Explore dataset information and class distribution
3. Drop leakage columns: `Laundering_type`, `Type`, `Is_Suspicious`
4. Conduct **stateless feature engineering** (computed per row, safe before split):
   - `Amount_log` — log-transform of transaction amount
   - `CurrencyMismatch` — payment vs. received currency mismatch flag
   - `CrossBorder` — sender vs. receiver bank location mismatch flag
   - Temporal features: `Hour`, `DayOfWeek`, `Month`, `IsWeekend`
5. Stratified split into train / validation / test sets **(70 / 15 / 15)**
6. Conduct **post-split account behavioral feature engineering** from the training set only:
   - `Sender_freq`, `Receiver_freq`, `Sender_txn_count`, `Receiver_txn_count`
   - Account mean and max amount statistics
7. Encode categorical variables using OrdinalEncoder (fit on train only)
8. Scale numerical features using StandardScaler (fit on train only)
9. Train Random Forest, XGBoost, and LightGBM with internal class imbalance handling
10. Tune classification threshold on the **validation set** using F2-Score
11. Evaluate final performance on the **test set** using the tuned threshold
12. Interpret the best model using SHAP
13. Benchmark inference performance for Edge AI deployment

### Methodology Notes

| Aspect | Decision |
|---|---|
| Target | `Is_laundering` |
| Leakage prevention | Drop `Laundering_type`, `Type`, `Is_Suspicious` before any modeling |
| Account ID handling | `Sender_account` and `Receiver_account` not encoded as raw IDs; converted to behavioral aggregates |
| Account feature engineering | Computed from train set only, then applied to validation/test |
| Stateless AML features | `Amount_log`, `CurrencyMismatch`, `CrossBorder`, `Hour`, `DayOfWeek`, `Month`, `IsWeekend` |
| Imbalance handling | `class_weight='balanced'` / `scale_pos_weight` — no oversampling |
| Split strategy | Stratified train / validation / test (70 / 15 / 15) |
| Threshold tuning | Optimized on validation set using F2-Score; test set for final evaluation only |
| Evaluation metrics | PR-AUC, ROC-AUC, Precision, Recall, F1, F2, confusion matrix, threshold report |

---

## Model Evaluation

### Classification Performance

All metrics are evaluated on the test set using the optimal threshold tuned on the validation set.

| Model | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Threshold |
|---|---|---|---|---|---|---|
| Random Forest | 0.4598 | 0.6293 | 0.5314 | 0.5985 | 0.9924 | 0.8556 |
| XGBoost | 0.7843 | 0.8251 | 0.8042 | 0.8608 | 0.9970 | 0.9700 |
| **LightGBM** | **0.8244** | **0.8211** | **0.8227** | **0.8666** | **0.9976** | **0.9771** |

> LightGBM is selected as the main XE-AML edge model because it achieves the highest PR-AUC and ROC-AUC while maintaining the strongest precision-recall balance. PR-AUC is emphasized over Accuracy because the dataset is highly imbalanced.

### Class Distribution
![Class Distribution](pencucian%20uang/pencucian%20uang/class_distribution.png)

### Amount Distribution
![Amount Distribution](pencucian%20uang/pencucian%20uang/amount_distribution.png)

### Confusion Matrices
![Confusion Matrices](pencucian%20uang/pencucian%20uang/confusion_matrices.png)

### Precision-Recall Curve
![Precision Recall Curve](pencucian%20uang/pencucian%20uang/precision_recall_curve.png)

### ROC Curve
![ROC Curve](pencucian%20uang/pencucian%20uang/roc_curve.png)

### Feature Importance
![Feature Importance](pencucian%20uang/pencucian%20uang/feature_importance_best_model.png)

---

## SHAP Explainability

SHAP is used to explain how the trained LightGBM model makes predictions. AML decisions must not only be accurate, but also transparent, auditable, and understandable for compliance officers and regulators. SHAP provides both global and local explanations for suspicious transaction detection.

Key features identified by SHAP include: transaction amount, sender transaction count, sender amount statistics, currency mismatch, and payment type.

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

## Inference Benchmark and Computational Performance

Computational performance is evaluated to determine whether the LightGBM model can operate efficiently on edge infrastructure. Low latency and small model size are particularly important because the proposed system aims to detect suspicious transactions before funds move further through the financial system.

The benchmark measures:

- **Inference latency** — time per transaction (ms)
- **Memory consumption** — RSS delta when loading the model (MB)
- **Model size** — serialized file size on disk (MB)
- **Transaction throughput** — predictions per second (TPS)
- **CPU usage** — normalized per logical core via background polling monitor

### Benchmark Environment

| Parameter | Value |
|---|---|
| Platform | Windows |
| CPU logical cores | 8 |
| CPU physical cores | 6 |
| RAM total | 7.71 GB |
| Python version | 3.12.9 |
| Warm-up inferences | 200 |
| Measurement samples | 1,000 single-row inferences |

### Benchmark Results — LightGBM

| Metric | Value | Interpretation |
|---|---:|---|
| **Mean Latency** | **1.2647 ms** | Average inference time per transaction |
| **Median Latency** | **1.1640 ms** | Typical inference time |
| **P95 Latency** | **1.7895 ms** | 95% of transactions processed within this limit (SLA target) |
| **P99 Latency** | **2.2710 ms** | Upper-bound near-real-time inference latency (SLA ceiling) |
| **Min / Max Latency** | **0.9042 / 2.8022 ms** | Observed latency range |
| **Model Size** | **1.015 MB** | Lightweight — suitable for edge deployment |
| **Memory Footprint** | **2.85 MB** | Additional RSS memory required when loading model + scaler |
| **Inference Memory Delta** | **4.70 MB** | Additional memory consumed during active inference loop |
| **Batch Throughput** | **169,072 TPS** | High-volume batch inference capability (5,000 samples) |
| **Stress Test Throughput** | **247,767 TPS** | Transaction processing capacity under sustained load (≥5 s loop) |
| **Avg CPU per-core** | **72.90%** | Normalized per logical core — valid measurement over 5-second loop |
| **CPU samples valid** | **48 / 48** | All monitor samples captured successfully |

> **Edge AI Suitability:** All six criteria passed ✅
> Model Size < 10 MB · Memory Footprint < 50 MB · Mean Latency < 5 ms · P99 Latency < 10 ms · Throughput > 1,000 TPS · Avg CPU per-core < 80%

### Notes on Measurement Methodology

**Memory Footprint** is measured as the RSS (Resident Set Size) delta before and after loading the model — not the total process memory. This isolates the actual memory cost attributable to the model itself.

**CPU Usage** is measured using `psutil` in a background polling thread (interval: 100 ms) during a sustained inference loop of at least 5 seconds. The raw value from `psutil.cpu_percent()` on Windows reports cumulative CPU across all cores (e.g., 8 cores × ~79% ≈ 583%). This value is normalized by dividing by the number of logical cores to obtain per-core utilization (583% ÷ 8 = **72.90% per-core**). A warm-up phase of 200 inferences is performed before measurement to eliminate JIT cold-start overhead.

**P95 and P99 latency** are more important than mean latency for production SLA assessment because they capture worst-case performance under realistic conditions.

### Raw Benchmark Output Screenshots

#### Model Size and Memory Footprint
![Model Size and Memory Footprint](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_model_size_memory.png)

#### Inference Latency
![Inference Latency](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_latency.png)

#### Batch Throughput
![Batch Throughput](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_batch_throughput.png)

#### Stress Test CPU and Memory
![Stress Test CPU and Memory](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_stress_cpu_memory.png)

#### Edge AI Suitability Assessment
![Edge AI Suitability Assessment](pencucian%20uang/pencucian%20uang/benchmark_outputs/benchmark_edge_ai_suitability.png)

---

## Repository Contents

```
README.md
.gitignore

pencucian uang/pencucian uang/
├── aml_detection_AML_FEATURE_ENGINEERING_FINAL.ipynb  ← main notebook (all sections)
├── SAML-D.csv                                          ← dataset (required before running)
│
├── Saved model artifacts
│   ├── aml_best_model.pkl          ← best trained model (LightGBM)
│   ├── aml_scaler.pkl              ← fitted StandardScaler
│   ├── aml_encoder.pkl             ← fitted OrdinalEncoder
│   ├── aml_threshold.pkl           ← optimal classification threshold
│   ├── aml_best_scores.pkl         ← best model evaluation scores
│   ├── aml_account_stats.pkl       ← account behavioral statistics (train set only)
│   ├── aml_feature_columns.pkl     ← final feature column list
│   ├── aml_numeric_medians.pkl     ← numeric medians for missing value imputation
│   ├── aml_cat_cols.pkl            ← categorical column list
│   ├── tmp_LightGBM.pkl            ← temporary LightGBM checkpoint
│   ├── tmp_XGBoost.pkl             ← temporary XGBoost checkpoint
│   └── tmp_Random_Forest.pkl       ← temporary Random Forest checkpoint
│
├── Evaluation results
│   ├── aml_model_comparison_results.csv
│   ├── threshold_report_all_models.csv
│   ├── threshold_report_lightgbm.csv
│   ├── threshold_report_xgboost.csv
│   └── threshold_report_random_forest.csv
│
├── Benchmark results
│   ├── inference_metrics_updated.json   ← latest benchmark results
│   ├── inference_metrics_refactored.json
│   └── inference_metrics.json
│
├── Visualization outputs
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

This implementation supports the innovation paper:

> **"Explainable Edge AI for Cost-Efficient and Trustworthy Anti-Money Laundering Systems"**
> Bella Adisty, Nela Eka Silvia Lumbanraja, Albert Jofrandi Hutapea
> Program Studi Informatika, Institut Teknologi Sains Bandung
> Innovation Paper Competition 2026 — Risk and Governance Summit

The proposed XE-AML framework combines:

- **Edge AI** for low-latency, near-real-time transaction screening before funds move further
- **LightGBM** for lightweight machine learning inference with small model footprint
- **SHAP** for explainable, auditable AML decision-making
- **Inference benchmarking** to evaluate deployment feasibility on edge infrastructure
- **Computational efficiency analysis** using latency, throughput, model size, memory, and CPU metrics

The computational evaluation supports the claim in Section 3.5 of the paper: LightGBM achieves a mean inference latency of **1.26 ms**, a model size of **1.015 MB**, a memory footprint of **2.85 MB**, and a transaction throughput of **169,072 TPS** — all within the thresholds required for efficient edge deployment.

---

## Notes

- The notebook is named `aml_detection_AML_FEATURE_ENGINEERING_FINAL.ipynb` and covers the complete pipeline from data loading (Section 1) through inference benchmarking (Section 10).
- The trained model (`aml_best_model.pkl`) and scaler (`aml_scaler.pkl`) are included because their combined size is under 2 MB.
- The notebook may not always render correctly on GitHub if the output is large. Result figures are provided separately as PNG files.
- CPU usage is reported as a per-core normalized value. Raw `psutil` output on multi-core Windows systems may show values above 100% (cumulative across all cores).
- Account behavioral features are computed after the train/validation/test split to prevent leakage. Statistics are derived from the training set only, then applied to validation and test sets.
