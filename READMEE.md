# 🛡️ The Retention Architect: Smart Churn Prediction System
### HackNU 2026 | Case: Higgsfield DS/ML
Team: The Grid Team

## 📋 Project Overview
Our solution is an end-to-end predictive pipeline designed to identify users at risk of leaving the platform. Unlike standard models, we distinguish between Voluntary Churn (active decision) and Involuntary Churn (payment failures/technical issues), allowing for targeted business interventions.

## 🚀 Key Features
* Dual-Classification: Separation of Active (Voluntary) vs. Passive (Involuntary) churn risks.
* Explainable AI (XAI): Integrated feature importance to justify why a user was flagged.
* Actionable Insights: Automated strategy generation for Product Managers.
* Performance: High Precision-Recall AUC and F1-score on synthetic user activity data.

## 🛠️ Tech Stack
* Language: Python 3.x
* Libraries: Pandas, NumPy, Scikit-learn, XGBoost/LightGBM
* Analysis: Matplotlib, Seaborn (Data Visualization)
* Explainability: SHAP / Lime (Model Interpretation)

## 📊 Methodology
1.  Data Engineering: Processed user activity logs, generation history, and payment metadata.
2.  Feature Selection: Identified key drivers such as "Time since last generation" and "Payment retry frequency."
3.  Model Training: Optimized a gradient-boosted classifier for maximum recall on high-value users.
4.  Categorization: Applied logic to separate technical failures from behavioral drops.

## 📉 Churn Drivers & Strategy
| Churn Type | Primary Drivers | Business Action |
| :--- | :--- | :--- |
| Voluntary | Low engagement, price sensitivity | Loyalty discounts, feature tutorials |
| Involuntary | Expired cards, API errors | Smart retry logic, UI notifications |

## ⚙️ Setup & Installation
```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)[your-username]/[repo-name].git

# Install dependencies
pip install -r requirements.txt

# Run the notebook/script
jupyter notebook churn_analysis.ipynb
