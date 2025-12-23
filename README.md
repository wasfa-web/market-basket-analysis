Market Basket Analysis Web App

This project implements a Market Basket Analysis web application using Streamlit and PySpark. It allows users to explore frequent itemsets and association rules from transaction data using the FP-Growth algorithm.

🔹 Features

Load and preprocess data:

Supports uploading a CSV file.

Cleans product names and removes duplicates.

Splits transactions into lists of items.

Frequent itemset mining:

Uses PySpark's FP-Growth algorithm.

Supports configurable minimum support and confidence.

Displays top N frequent itemsets and association rules.

Visualization:

Bar charts and pie charts for frequent itemsets.

Bubble charts and network graphs for association rules.

Interactive Streamlit dashboard.

Export options:

Download frequent itemsets and association rules as CSV.

Download cleaned transaction data.

Business insights:

Highlights strong association rules based on lift and confidence.

Provides actionable recommendations (e.g., bundling products).

📁 Dataset

The analysis was originally performed on the full Retail Transactions Dataset (link below).

For testing or demonstration, a smaller CSV can be uploaded via the Upload CSV option.

Full dataset: Kaggle Retail Transactions Dataset

Sample dataset: Provided within the repository (retail_transactions_sample.csv) or upload your own.

⚙️ Usage

Run the Streamlit app:

streamlit run app.py


Configure Min Support, Min Confidence, and number of top itemsets/rules via the sidebar.

Upload a CSV file if desired.

Explore tabs:

📦 Frequent Itemsets

🔗 Association Rules

📈 Visualizations

📄 Export Results

💡 Business Insights

📊 Dashboard Overview

📄 Export Cleaned Data

🔧 Notes

Default Spark configuration assumes HDFS path for the large dataset.

Sample dataset is used for quick testing with smaller min support/confidence values.

For the full dataset, adjust min support/confidence to avoid too many frequent itemsets.
## How to Run the Project
1. Install required libraries:
```bash
pip install pyspark streamlit pandas plotly matplotlib networkx
