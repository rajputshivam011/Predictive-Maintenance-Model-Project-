# AI-Powered Predictive Maintenance System for Manufacturing Equipment Using Machine Learning

A machine-learning project that analyzes machine operating conditions to predict potential failures and support data-driven maintenance planning. The project uses the **AI4I 2020 Predictive Maintenance Dataset** and combines exploratory data analysis, classification models, model explainability, and Power BI dashboards.

> **Project focus:** Predictive maintenance for manufacturing equipment. SQL is intentionally outside the scope of this main project.

## Business Problem

Unexpected machine failures can interrupt production, increase maintenance costs, and affect delivery performance. This project investigates whether machine operating data can help identify potential failures so maintenance teams can prioritize inspections and plan interventions. It is a proof of concept; operational savings must be validated in a real manufacturing environment.

## Objectives

- Explore machine operating data and patterns associated with recorded failures.
- Build and evaluate machine-failure classification models.
- Compare models using accuracy, precision, recall, F1-score, ROC-AUC, and average precision where available.
- Account for class imbalance when interpreting performance.
- Explain model behavior using feature importance and SHAP.
- Translate analytical findings into manufacturing-oriented maintenance insights.
- Present key findings through Power BI dashboards.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — 10,000 records and 14 original columns. The classification target is `machine_failure`.

| Target class | Records | Percentage |
|---|---:|---:|
| Normal operation (`0`) | 9,661 | 96.61% |
| Machine failure (`1`) | 339 | 3.39% |
| **Total** | **10,000** | **100%** |

Recorded failures account for only 3.39% of observations. Accuracy alone is therefore insufficient for model selection; recall, precision, F1-score, and other relevant metrics must also be considered.

### Main model features

- `product_type`
- `air_temperature_k`
- `process_temperature_k`
- `rotational_speed_rpm`
- `torque_nm`
- `tool_wear_min`

**Target:** `machine_failure`

The dataset also contains failure-mode indicators such as TWF, HDF, PWF, OSF, and RNF for analysis. Outcome-related indicators should not be used as model inputs when they directly encode failure outcomes, as this can introduce target leakage.

## Project Workflow

1. **Data loading and preparation** — inspect the dataset, rename columns, review data types, and check missing values and duplicates.
2. **Exploratory data analysis (EDA)** — examine target imbalance, failure types, feature distributions, summary statistics, and correlations.
3. **Bivariate analysis** — compare operating parameters under normal and failure conditions and interpret findings from a manufacturing perspective.
4. **Model development** — train classification algorithms using selected operating features.
5. **Hyperparameter tuning** — use cross-validation and GridSearchCV for selected models.
6. **Evaluation and comparison** — assess confusion matrices, accuracy, precision, recall, F1-score, ROC-AUC, and average precision as applicable.
7. **Model interpretation** — review tree-based feature importance and SHAP explanations.
8. **Risk analysis and maintenance recommendations** — examine predicted risk and translate findings into practical inspection and monitoring suggestions.
9. **Power BI reporting** — communicate machine health, failure analysis, operating parameters, and model/maintenance insights.

## Models Evaluated

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

Hyperparameter tuning is applied where appropriate. Because the target is imbalanced, models are compared using more than overall accuracy.

## Key Results

The selected Decision Tree configuration was:

`DecisionTreeClassifier(max_depth=10, min_samples_leaf=4, random_state=42)`

### Test-set performance

| Metric | Tuned Decision Tree |
|---|---:|
| Accuracy | 98.55% |
| Precision | 79.10% |
| Recall | 77.94% |
| F1-score | 78.52% |
| ROC-AUC | 91.84% |
| Average Precision (AP) | 75.36% |

### Confusion matrix

| | Predicted normal | Predicted failure |
|---|---:|---:|
| **Actual normal** | 1,918 | 14 |
| **Actual failure** | 15 | 53 |

On the reported test set, the model correctly identified 53 of 68 failure observations, missed 15 failures, and produced 14 false-positive alerts. In a real factory, the acceptable balance between missed failures and unnecessary inspections depends on failure severity, inspection cost, and operational constraints.

A tuned Gradient Boosting model achieved 98.60% accuracy, 84.48% precision, 72.06% recall, and a 77.78% F1-score in the recorded comparison. The Decision Tree was selected as the preferred model in that comparison because it provided a stronger recall/F1 balance while remaining relatively interpretable.

*These figures reflect the project's recorded evaluation run. They do not guarantee performance on other machines, factories, or future operating conditions.*

## Explainability and Maintenance Insights

Tree-based feature importance and SHAP analysis were used to understand which inputs most influenced model predictions. In the recorded analysis, **torque** was the leading feature, followed by air temperature and tool wear in the Decision Tree feature-importance results. SHAP global importance also placed torque prominently.

These findings can guide further investigation—for example, reviewing torque patterns, tool-wear history, and operating conditions when prioritizing machine inspections. They do **not** establish that a feature causes a failure. Engineering validation and additional production data are needed before changing maintenance policy.

## Power BI Dashboard

Power BI is used to communicate findings through dashboard views covering:

- **Machine Health Overview** — total records, normal observations, failure observations, and failure rate.
- **Failure Analysis** — recorded failure patterns and failure types.
- **Operating Parameter Analysis** — machine operating conditions and their relationship to failures.
- **Model Insights & Maintenance** — model findings and maintenance-oriented insights.

If dashboard screenshots are added to the repository, place them in a folder such as `reports/images/` and embed them here. Example:

```markdown
![Machine Health Overview](reports/images/machine-health-overview.png)
```

## Technology Stack

- Python
- Jupyter Notebook
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- SHAP
- Power BI

## Suggested Repository Structure

Adapt this example to match the files actually uploaded:

```text
predictive-maintenance/
├── data/
│   └── raw/
│       └── ai4i2020.csv
├── notebooks/
│   └── PREDICTIVE MAINTENENCE PROJECT.ipynb
├── reports/
│   ├── Predictive_Maintenance_Final_Report_With_PowerBI_Images.docx
│   └── images/
│       └── (optional dashboard screenshots)
├── README.md
└── requirements.txt
```

This is a suggested layout, not a claim that every listed file is already present. Update filenames and folders to match your repository. Include appropriate dataset attribution and comply with the dataset provider's terms.

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create and activate a virtual environment (recommended)

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Otherwise, install the main libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
```

### 4. Add the dataset

Place `ai4i2020.csv` in `data/raw/` if using the suggested folder layout. Check the dataset path at the beginning of the notebook and update it if your folder structure differs.

### 5. Run the notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells in order. Results can vary if the data split, features, package versions, random state, or tuning configuration changes.

## Limitations

- The dataset is a benchmark dataset and may not represent every real manufacturing environment.
- Failure events are rare, so accuracy alone can give a misleading impression of model quality.
- A false negative may mean a failure is missed; a false positive may trigger an unnecessary inspection. These errors have different operational costs.
- SHAP values and feature importance explain model behavior; they do not prove causation.
- The model is a decision-support prototype, not a replacement for engineering judgment, safety procedures, or a validated maintenance system.
- Actual improvements in downtime, cost, and equipment availability have not been established by this dataset analysis alone.

## Future Improvements

- Validate the model using data from real production equipment and time-based holdout testing.
- Calibrate risk thresholds based on the cost of missed failures versus unnecessary inspections.
- Monitor model performance and data drift after deployment.
- Integrate predictions into a maintenance workflow with human review and documented outcomes.
- Extend dashboard reporting with validated operational KPIs once real deployment data is available.

👨‍💻 Author 

Shivam Kr. Rajput

*Predictive Maintenance Model| Manufacturing Analytics | Machine Learning | Smart Manufacturing*


Connect With Me LinkedIn: www.linkedin.com/in/shivamkrrajput GitHub: https://github.com/rajputshivam011

⭐ If you found this project useful, consider giving it a Star on GitHub and sharing your feedback!


