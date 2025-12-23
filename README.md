Market Basket Analysis Web App
Project Overview

Interactive web application for analyzing retail transactions using FP-Growth with PySpark.
The app extracts frequent itemsets and association rules to help businesses make decisions on product bundling, promotions, and insights.

Features

Load and clean transactional data (remove duplicates, handle missing values, standardize product names)

FP-Growth for mining frequent itemsets and association rules (support, confidence, lift)

Interactive dashboard with Streamlit

Visualizations:

Bar charts (Top Frequent Items)

Pie charts (Top Itemsets)

Bubble charts (Confidence vs Lift)

Network graphs (Association Rules)

Export results and cleaned data as CSV

Business insights with actionable recommendations

Technologies Used

Python

PySpark

Streamlit

Pandas

Plotly

Matplotlib

NetworkX

Dataset

Sample dataset (included for quick testing): retail_transactions_sample.csv

Full dataset (large, optional): Retail Transactions Dataset on Kaggle

Note: The code can run with the sample dataset for review. Use the full dataset for complete analysis. Adjust support/confidence parameters to avoid too many frequent itemsets.

How to Run

Clone the repository:

git clone <repo-url>
cd market-basket-analysis


Install dependencies:

pip install pyspark streamlit pandas plotly matplotlib networkx


Run the app:

streamlit run app.py


Configure parameters in the sidebar:

Min Support

Min Confidence

Top N Frequent Itemsets / Association Rules

Upload your CSV (optional)

Explore tabs:

📦 Frequent Itemsets

🔗 Association Rules

📈 Visualizations

📄 Export Results

💡 Business Insights

📊 Dashboard Overview

📄 Export Cleaned Data

Notes

Optimized for large datasets via PySpark; sample dataset used for quick testing.

Adjust min support and min confidence when running full dataset to control output size.

Visualizations and metrics are limited to safe subsets for memory efficiency.
