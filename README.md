# DemandPulse AI
## Real-Time Demand Forecasting & Supply Chain Optimization System

An end-to-end **Data Science and Machine Learning project** designed to analyze retail sales patterns, predict stockout risks, understand inventory behavior, and support smarter inventory and supply chain decisions.

> **Business Question:** How can businesses predict stockout risks, understand inventory pressure, and make smarter restocking decisions before products become unavailable?

---

## Project Overview

Retail businesses frequently face challenges related to inventory management, unpredictable customer demand, and product stockouts.

Excess inventory increases storage costs, while insufficient inventory leads to lost sales, customer dissatisfaction, and operational inefficiencies.

DemandPulse AI focuses on transforming retail transaction and inventory data into actionable insights using:

**Retail Analytics + Feature Engineering + Inventory Analysis + Machine Learning + Stockout Risk Prediction + Supply Chain Optimization**

The system analyzes historical retail data and identifies patterns that can help businesses make better inventory planning decisions.

---

## Business Problem

The system focuses on answering the following questions:

* Which products have a high risk of stockouts?
* Which products require urgent restocking?
* How does customer demand affect inventory levels?
* Which factors contribute to inventory pressure?
* How do promotions influence retail sales?
* How can businesses reduce inventory shortages?
* How can businesses improve inventory planning?
* Which products require closer monitoring?

---

## Project Objectives

* Analyze large-scale retail transaction data.
* Perform data cleaning and validation.
* Understand retail sales and inventory patterns.
* Perform feature engineering on retail data.
* Identify important inventory-related indicators.
* Analyze sales, revenue, and profitability.
* Predict stockout risk using Machine Learning.
* Identify products with high inventory pressure.
* Support smarter restocking decisions.
* Develop a data-driven inventory optimization system.

---

# Dataset Overview

The project uses a synthetic Retail POS (Point of Sale) dataset containing approximately **1 million records**.

The dataset represents retail transaction and inventory-related information for analyzing sales behavior and stockout risks.

### Dataset Files

| File Name | Description |
|---|---|
| `demandpulse_ai.csv` | Original retail dataset |
| `demandpulse_ai_clean.csv` | Cleaned retail dataset |
| `demandpulse_ai_clean.pkl` | Serialized cleaned dataset |
| `demandpulse_data_quality_report.csv` | Data quality analysis report |

### Dataset Characteristics

- Approximately 1,000,000 records.
- Retail transaction-level information.
- Sales and revenue-related variables.
- Inventory and stock-related variables.
- Pricing and promotional information.
- Time and calendar-related features.
- Stockout risk classification target.

---

# Machine Learning Objective

### Prediction Type

**Binary Classification**

### Target Variable

```python
stockout_target
```

The target variable identifies the stockout risk associated with retail inventory.

| Target | Description | Records |
|---|---|---:|
| 0 | Low Stockout Risk | 840,000 |
| 1 | High Stockout Risk | 160,000 |

### Target Distribution

```text
Low Stockout Risk   : 840,000 (84%)
High Stockout Risk  : 160,000 (16%)
```

The classification task aims to identify inventory situations associated with high stockout risk.

---

# Machine Learning Workflow

```text
Retail POS Dataset
        ↓
Data Understanding
        ↓
Data Cleaning & Validation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Inventory & Sales Analysis
        ↓
Target Variable Preparation
        ↓
Train/Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Stockout Risk Prediction
        ↓
Inventory Optimization
        ↓
Business Recommendations
```

---

# Feature Engineering

Feature engineering plays an important role in DemandPulse AI.

The project creates additional variables from existing retail, inventory, sales, and profitability information to help identify inventory pressure and stockout patterns.

### Engineered Features

| Feature | Description |
|---|---|
| `day` | Day of the month |
| `day_of_week_num` | Numerical representation of the weekday |
| `quarter` | Calendar quarter |
| `is_weekend` | Identifies weekend transactions |
| `sales_per_transaction` | Sales generated per transaction |
| `revenue_per_unit` | Revenue generated per unit |
| `stock_consumption_ratio` | Inventory consumption indicator |
| `stock_to_sales_ratio` | Relationship between available stock and sales |
| `inventory_change` | Change in inventory |
| `stock_gap` | Inventory shortage indicator |
| `lead_time_stock_pressure` | Inventory pressure considering supplier lead time |
| `restock_urgency` | Restocking urgency indicator |
| `purchase_gap` | Purchase gap indicator |
| `profit_per_unit` | Profit generated per unit |
| `profit_margin_pct` | Profit margin percentage |
| `discount_amount` | Discount amount |
| `promotion_sales_effect` | Promotion-related sales impact |
| `has_previous_stockout` | Indicates previous stockout occurrence |

These engineered features help the system understand the relationship between retail demand, inventory availability, and stockout risk.

---

# Inventory & Sales Analysis

DemandPulse AI focuses on understanding the relationship between retail sales and available inventory.

### Sales Analysis

Analyzes:

* Sales patterns.
* Revenue generation.
* Sales per transaction.
* Revenue per unit.
* Promotional sales behavior.

### Inventory Analysis

Analyzes:

* Inventory consumption.
* Stock availability.
* Stock-to-sales relationship.
* Inventory changes.
* Inventory gaps.
* Restocking urgency.

### Supply Chain Analysis

Analyzes:

* Supplier lead-time pressure.
* Inventory shortage indicators.
* Purchase gaps.
* Previous stockout patterns.

---

# Stockout Risk Prediction

The primary Machine Learning objective is to classify inventory situations into two categories.

### Low Stockout Risk

Identifies inventory situations with a lower predicted stockout risk.

### High Stockout Risk

Identifies inventory situations that require closer attention because of higher predicted stockout risk.

The prediction system aims to support proactive inventory monitoring instead of relying only on historical stock reports.

---

# Machine Learning

## Prediction Model

**Binary Classification**

The project uses Scikit-learn for developing and evaluating Machine Learning models.

### Target

```python
stockout_target
```

### Model Development Strategy

The model development process includes:

* Feature preparation.
* Target variable separation.
* Training and testing data preparation.
* Model training.
* Model evaluation.
* Feature importance analysis.
* Model optimization.

### Model Evaluation Metrics

The following metrics are relevant for evaluating the classification models:

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correctness of high-risk predictions |
| Recall | Ability to identify high-risk cases |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish between classes |
| Confusion Matrix | Detailed classification performance |

**Important:** Since the dataset contains an imbalanced target distribution (84% vs 16%), Precision, Recall, F1 Score, and ROC-AUC are particularly important alongside accuracy.

---

# Business Decision Engine

The ultimate objective is to convert Machine Learning predictions into useful inventory decisions.

| Inventory Condition | Recommended Action |
|---|---|
| High stockout risk | Prioritize inventory monitoring |
| High restock urgency | Review replenishment requirements |
| Low stockout risk | Continue regular inventory monitoring |
| High lead-time pressure | Review supplier replenishment planning |
| Previous stockout detected | Monitor inventory availability closely |
| Promotion-related sales increase | Review inventory requirements |

The purpose is not simply to predict stockout risk, but to support better inventory and supply chain decisions.

---

# Final Demand Intelligence System

```text
Historical Retail Data
        ↓
Sales & Inventory Analysis
        ↓
Feature Engineering
        ↓
Machine Learning Model
        ↓
Stockout Risk Prediction
        ↓
Inventory Risk Assessment
        ↓
Restocking Recommendations
        ↓
Supply Chain Optimization
```

The system is designed to generate:

```text
Stockout Risk Classification
Inventory Pressure Indicators
Restocking Urgency Insights
Sales & Inventory Analysis
Business Recommendations
```

---

# Key Business Insights

### Inventory Risk Identification

Identifies inventory situations with high stockout risk, helping businesses prioritize inventory monitoring.

### Demand & Sales Understanding

Analyzes sales behavior and promotional effects to understand factors influencing retail demand.

### Inventory Management

Uses stock-related indicators to understand inventory availability and potential shortages.

### Restocking Opportunities

Identifies situations where inventory pressure and replenishment urgency require attention.

### Supply Chain Efficiency

Supports better inventory planning by considering stock availability and supplier lead-time pressure.

### Strategic Benefit

The project moves retail inventory management from:

```text
Historical Retail Reporting
        ↓
Predictive Inventory Intelligence
        ↓
Stockout Risk Assessment
        ↓
Data-Driven Business Decisions
```

---

# Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**
* **Streamlit**

# How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/DemandPulse-AI.git
```

## 2. Navigate to the Project Directory

```bash
cd DemandPulse-AI
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the Project Notebook

```text
Real-Time Demand Forecasting & Supply Chain Optimization System.ipynb
```

Run the notebook cells sequentially to reproduce the data analysis, feature engineering, and Machine Learning workflow.

---

# Future Scope

* Advanced demand forecasting models.
* Improved stockout risk prediction.
* Model optimization and hyperparameter tuning.
* Inventory replenishment optimization.
* Automated reorder recommendations.
* Flask/FastAPI prediction API.
* Streamlit interactive inventory dashboard.
* Power BI business intelligence dashboard.
* Real-time inventory monitoring.
* Automated low-stock alerts.
* Supplier performance analysis.
* Model monitoring and retraining.

---

# Project Outcome

DemandPulse AI demonstrates an end-to-end **Retail Data Science and Machine Learning workflow** focused on demand intelligence, stockout risk identification, and inventory optimization.

It combines:

**Data Engineering + Data Cleaning + Exploratory Data Analysis + Feature Engineering + Inventory Analytics + Machine Learning + Business Decision Making**

to address an important retail business problem:

> **How can businesses identify stockout risks, understand inventory pressure, and make smarter restocking decisions using data and Machine Learning?**

---

# Author

**Shivam Shrivastav**

BCA Student | Data Analytics & Machine Learning

If you find this project useful, consider giving the repository a ⭐.
