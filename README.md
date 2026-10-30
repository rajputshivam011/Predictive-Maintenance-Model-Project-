# AI-Powered Predictive Maintenance System for Manufacturing Equipment

A machine-learning project that predicts whether an industrial milling machine will fail, using routine operating readings, so maintenance can be planned before a breakdown instead of after it.

## Problem

Unplanned machine failures cause downtime, higher maintenance cost, delivery delays and lower OEE. This project asks: *can torque, speed, temperature and tool wear tell us a machine is about to fail, and which of these readings matter most?*

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (Matzka, 2020): synthetic data modelled on a real milling machine.

- 10,000 records, 14 columns, no missing values, no duplicates
- 339 failures (**3.39%**), so the problem is highly imbalanced
- Predictors used: product type, air temperature, process temperature, rotational speed, torque, tool wear
- Target: `Machine failure`
- The failure-mode flags (TWF, HDF, PWF, OSF, RNF) were used for analysis only, not as predictors, because they define the target

## Workflow

1. Data understanding and quality checks
2. Exploratory and bivariate analysis
3. Data preparation: label encoding, stratified 80/20 split (random_state 42), scaling where needed
4. Six models, tuned with `GridSearchCV` (F1 objective)
5. Evaluation on failure-class precision, recall, F1, ROC and precision-recall curves
6. Explainability with SHAP
7. Root-cause analysis by failure mode and maintenance recommendations
8. Saved model, reusable prediction function and a Power BI dashboard

## Results

Test set: 2,000 records, 68 of them failures.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression* | 96.80% | 0.67 | 0.12 | 0.20 |
| KNN* | 97.25% | 0.81 | 0.25 | 0.38 |
| **Tuned Decision Tree** | **98.55%** | **0.79** | **0.78** | **0.79** |
| Random Forest | 98.50% | 0.88 | 0.65 | 0.75 |
| Tuned SVM | 98.05% | 0.75 | 0.63 | 0.69 |
| Tuned Gradient Boosting | 98.60% | 0.84 | 0.72 | 0.78 |

\* Baseline results. Tuned KNN (K=5): precision 0.83, recall 0.29, F1 0.43.

**Selected model: tuned Decision Tree** (Gini, `max_depth=10`, `min_samples_leaf=4`). It has the best recall and F1 and is easy to explain.

- Caught 53 of 68 failures, with 14 false alarms and 15 missed failures
- ROC-AUC 0.918, average precision 0.754 (failure prevalence is 0.034)

## Key findings

- Failed machines ran at higher torque (about 50.2 Nm vs 39.6 Nm) and higher tool wear (about 144 min vs 107 min). Temperature differences were small.
- SHAP ranks torque first by a wide margin, then tool wear.
- Each failure mode has its own operating signature, so maintenance should be targeted by mode.
- L-type products had the highest failure rate (3.92%).

## Tech stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SHAP, Joblib, Jupyter Notebook, Power BI

## Repository structure

```
.
├── PREDICTIVE_MAINTENENCE_PROJECT.ipynb   # full analysis and modelling
├── data/
│   └── ai4i2020.csv                       # dataset (download from UCI)
├── models/
│   └── predictive_maintenance_dt.pkl      # saved tuned Decision Tree
├── dashboard/                             # Power BI file and screenshots
├── reports/
│   ├── Predictive_Maintenance_Project_Report.docx
│   └── Predictive_Maintenance_Presentation.pptx
└── README.md
```

Adjust the paths to match your repository.

## How to run

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install numpy pandas matplotlib seaborn scikit-learn shap joblib jupyter
jupyter notebook PREDICTIVE_MAINTENENCE_PROJECT.ipynb
```

Place `ai4i2020.csv` where the notebook expects it. The notebook loads it from the working directory. Some cells also use local Windows paths for the saved model and dashboard images, so update those to your own folders.

## Using the saved model

```python
import joblib
import pandas as pd

model = joblib.load("models/predictive_maintenance_dt.pkl")

# product_type encoding: H = 0, L = 1, M = 2
machine = pd.DataFrame([{
    "product_type": 1,
    "air_temperature_k": 302.0,
    "process_temperature_k": 311.5,
    "rotational_speed_rpm": 1400,
    "torque_nm": 55.0,
    "tool_wear_min": 220,
}])

print(model.predict(machine))              # 1 = failure predicted
print(model.predict_proba(machine)[:, 1])  # failure probability
```

Keep the six features in this order. A predicted failure means "inspect this machine", not a confirmed fault.

## Limitations

- Synthetic dataset, so performance on real plant data may differ
- Only 68 failures in the test set, so small differences between models are within noise
- 15 of 68 failures are still missed
- Decision Tree probabilities are not calibrated
- Six sensor features only; no vibration or maintenance history

## Future work

- Validate on real machine data
- Class weighting and decision-threshold tuning to raise recall
- Add vibration and maintenance-history features
- Live monitoring and integration with maintenance workflows

## Reference

Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), pp. 69-74. IEEE.


👨‍💻 Author Shivam Kr. Rajput

Predictive Maintenance Model

Connect With Me LinkedIn: www.linkedin.com/in/shivamkrrajput GitHub: https://github.com/rajputshivam011

⭐ If you found this project useful, consider giving it a Star on GitHub and sharing your feedback!
