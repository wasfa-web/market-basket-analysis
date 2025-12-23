Market Basket Analysis Web App

This project implements a Market Basket Analysis web application using Streamlit and PySpark. It allows users to explore frequent itemsets and association rules from transaction data using the FP-Growth algorithm.

🔹 Features
Load and Preprocess Data

Supports uploading a CSV file.

Cleans product names and removes duplicates.

Splits transactions into lists of items for analysis.

Frequent Itemset Mining

Uses PySpark's FP-Growth algorithm.

Configurable minimum support and confidence.

Displays top N frequent itemsets and association rules.

Visualization

Bar charts and pie charts for frequent itemsets.

Bubble charts and network graphs for association rules.

Fully interactive Streamlit dashboard.

Export Options

Download frequent itemsets and association rules as CSV.

Download cleaned transaction data.

Business Insights

Highlights strong association rules based on lift and confidence.

Provides actionable recommendations (e.g., product bundling, promotions).

📁 Dataset

The analysis was originally performed on the full Retail Transactions Dataset:

Full dataset (Kaggle): Retail Transactions Dataset:https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset?resource=download

For testing or demonstration, a smaller CSV can be uploaded via the Upload CSV option in the app.

Sample dataset: Provided within the repository (retail_transactions_sample.csv) or upload your own.

⚙️ Usage

Run the Streamlit app:

streamlit run app.py


Configure parameters via the sidebar:

Min Support

Min Confidence

Top N Frequent Itemsets / Association Rules

Upload a CSV file if desired.

Explore the following tabs:

📦 Frequent Itemsets

🔗 Association Rules

📈 Visualizations

📄 Export Results

💡 Business Insights

📊 Dashboard Overview

📄 Export Cleaned Data

🔧 Notes

Default Spark configuration assumes HDFS path for the large dataset.

The sample dataset is used for quick testing with smaller min support/confidence values.

For the full dataset, adjust min support/confidence to avoid too many frequent itemsets.
