
# Market Basket Analysis Web App

## Project Overview

Interactive web application for analyzing retail transactions using **FP-Growth** with **PySpark**.
The app extracts **frequent itemsets** and **association rules** to help businesses make decisions on **product bundling, promotions, and insights**.

---

## Features

* Load and clean transactional data (remove duplicates, handle missing values, standardize product names)
* FP-Growth for mining frequent itemsets and association rules (support, confidence, lift)
* Interactive dashboard with **Streamlit**
* Visualizations:

  * Bar charts (Top Frequent Items)
  * Pie charts (Top Itemsets)
  * Bubble charts (Confidence vs Lift)
  * Network graphs (Association Rules)
* Export results and cleaned data as CSV
* Business insights with actionable recommendations

---

## Technologies Used

* Python
* PySpark
* Streamlit
* Pandas
* Plotly
* Matplotlib
* NetworkX

---

## Dataset

* **Sample dataset (included for quick testing):** `retail_transactions_sample.csv`
* **Full dataset (used for complete analysis):** [Retail Transactions Dataset on Kaggle](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset?resource=download)

The full analysis was performed on a retail dataset containing **1,000,000 transactions across 81 unique products**. The repository includes a smaller sample dataset for quick testing, while the full dataset is linked separately due to its size.

> **Note:** The code can run with the sample dataset for review. Use the full dataset for complete analysis and adjust support/confidence parameters as needed to control the number of frequent itemsets and association rules.
---

## How to Run

1. Clone the repository:

```bash
git clone <repo-url>
cd market-basket-analysis
```

2. Install dependencies:

```bash
pip install pyspark streamlit pandas plotly matplotlib networkx
```

3. Run the app:

```bash
streamlit run app.py
```

4. Configure parameters in the sidebar:

* Min Support
* Min Confidence
* Top N Frequent Itemsets / Association Rules
* Upload your CSV (optional)

5. Explore tabs:

* 📦 Frequent Itemsets
* 🔗 Association Rules
* 📈 Visualizations
* 📄 Export Results
* 💡 Business Insights
* 📊 Dashboard Overview
* 📄 Export Cleaned Data

---

## Notes

* Optimized for large datasets via PySpark; sample dataset used for quick testing.
* Adjust **min support** and **min confidence** when running full dataset to control output size.
* Visualizations and metrics are limited to safe subsets for memory efficiency.
