import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, array_distinct, udf
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.types import ArrayType, StringType
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

# Streamlit UI config
st.set_page_config(page_title=" Market Basket Analysis", layout="wide")

# Start Spark session
spark = SparkSession.builder \
    .appName("MarketBasketAnalysis") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

# Format float values for display
def format_float(val):
    if val is None:
        return "N/A"
    return f"{val:.4f}".rstrip("0").rstrip(".")

# Clean items UDF
def clean_items(item_list):
    return [
        i.lower().strip()
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
        for i in item_list if i
    ]

clean_items_udf = udf(clean_items, ArrayType(StringType()))

# Preprocess data
def preprocess_data(df):
    df = df.select("Product").dropna()
    df = df.withColumn("items", split(col("Product"), ","))
    df = df.withColumn("items", clean_items_udf(col("items")))
    df = df.withColumn("items", array_distinct(col("items")))
    return df.select("items")

# Show raw data
def display_raw_data(df):
    df_sample = df.limit(100).toPandas()
    st.subheader("🧾 Raw Transaction Data (First 100 rows)")
    st.dataframe(df_sample)

# Sidebar - Dataset
st.sidebar.header("📁 Dataset Options ")
file_path = "hdfs://localhost:9000/user/vboxuser/data/Retail_Transactions_Dataset.csv"
upload_file = st.sidebar.file_uploader("📄 Or upload your CSV file", type="csv")

if upload_file:
    dataset_path = f"/tmp/{upload_file.name}"
    with open(dataset_path, "wb") as f:
        f.write(upload_file.getbuffer())
    default_support, default_confidence = 0.5, 0.5
else:
    dataset_path = file_path
    default_support, default_confidence = 0.002, 0.03

# Sidebar - Parameters
st.sidebar.header("⚙️ Model Parameters")
min_support = st.sidebar.number_input("📉 Min Support", 0.0001, 1.0, value=default_support, step=0.0001, format="%.3g")
min_confidence = st.sidebar.number_input("📈 Min Confidence", 0.0001, 1.0, value=default_confidence, step=0.0001, format="%.3g")
top_n_items = st.sidebar.slider("🔝 Top N Frequent Itemsets", 1, 50, 10)
top_n_rules = st.sidebar.slider("🔝 Top N Association Rules", 1, 50, 10)

if "reset" not in st.session_state:
    st.session_state.reset = False
if st.sidebar.button("🚀 Start Analysis"):
    st.session_state.reset = False
    st.session_state.run = True
    st.session_state["top_n_items"] = top_n_items
    st.session_state["top_n_rules"] = top_n_rules
if st.sidebar.button("🔄 Reset"):
    st.session_state.clear()

# Load and clean data
def load_data():
    if upload_file:
        df = spark.read.csv(dataset_path, header=True, inferSchema=True)
    else:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
    display_raw_data(df)
    return preprocess_data(df)

# FP-Growth model
def run_fpgrowth(df, support, confidence):
    fp = FPGrowth(itemsCol="items", minSupport=support, minConfidence=confidence)
    model = fp.fit(df)
    rules_df = model.associationRules
    rules_df = rules_df.withColumn("antecedent", clean_items_udf(col("antecedent")))
    rules_df = rules_df.withColumn("consequent", clean_items_udf(col("consequent")))
    rules_df = rules_df.filter(col("antecedent") != col("consequent"))
    return model.freqItemsets.orderBy("freq", ascending=False), rules_df.orderBy("lift", ascending=False)

# Visualization functions
def plot_bar_chart(freq_df):
    df = freq_df.toPandas()
    df["items"] = df["items"].apply(lambda x: ", ".join(x))
    df["freq"] = df["freq"].apply(format_float)
    df_sorted = df.sort_values(by="freq", ascending=False).head(top_n_items)
    fig = px.bar(df_sorted, x='freq', y='items', orientation='h')
    fig.update_layout(yaxis=dict(autorange="reversed"), title="Top Frequent Items", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

def plot_pie_chart(freq_df):
    df = freq_df.limit(10).toPandas()
    df["items"] = df["items"].apply(lambda x: ", ".join(x))
    fig = px.pie(df, values="freq", names="items", title="Top Frequent Itemsets (Pie Chart)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_bubble_chart(rules_df):
    df = rules_df.toPandas()
    if df.empty:
        st.warning("No rules to display.")
        return
    df["antecedent"] = df["antecedent"].apply(lambda x: ", ".join(x))
    fig = px.scatter(df, x="confidence", y="lift", size="lift", color="antecedent", title="Bubble Chart: Confidence vs Lift")
    st.plotly_chart(fig, use_container_width=True)

def plot_network_graph(rules_df):
    df = rules_df.toPandas()
    if df.empty:
        st.warning("No rules to visualize.")
        return
    G = nx.DiGraph()
    for _, row in df.head(15).iterrows():
        a = ", ".join(row["antecedent"]) if isinstance(row["antecedent"], list) else row["antecedent"]
        c = ", ".join(row["consequent"]) if isinstance(row["consequent"], list) else row["consequent"]
        G.add_edge(a, c)
    pos = nx.spring_layout(G, seed=42, k=1.3)
    node_sizes = [3000 + len(node) * 100 for node in G.nodes()]
    fig, ax = plt.subplots(figsize=(13, 8), facecolor="#111111")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#90EE90", node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="black")
    ax.set_title("Network Graph of Association Rules", color="white")
    ax.axis("off")
    st.pyplot(fig)

# Run analysis
if st.session_state.get("run", False):
    with st.spinner("Running market basket analysis..."):
        df = load_data()
        freq_df, rules_df = run_fpgrowth(df, min_support, min_confidence)
        st.session_state["df"] = df
        st.session_state["freq_df"] = freq_df
        st.session_state["rules_df"] = rules_df

# Tabs
if "freq_df" in st.session_state and "rules_df" in st.session_state:
    df = st.session_state["df"]
    freq_df = st.session_state["freq_df"]
    rules_df = st.session_state["rules_df"]

    tabs = st.tabs([
        "📦 Frequent Itemsets",
        "🔗 Association Rules",
        "📈 Visualizations",
        "📄 Export Results",
        "💡 Business Insights",
        "📊 Dashboard Overview",
        "📄 Export Cleaned Data"
    ])

    with tabs[0]:
        st.header("📦 Frequent Itemsets")
        df_top = freq_df.limit(st.session_state["top_n_items"]).toPandas()
        df_top["items"] = df_top["items"].apply(lambda x: ", ".join(x))
        df_top["freq"] = df_top["freq"].apply(format_float)
        st.dataframe(df_top)

    with tabs[1]:
        st.header("🔗 Association Rules")
        df_rules_display = rules_df.limit(st.session_state["top_n_rules"]).toPandas()
        df_rules_display["confidence"] = df_rules_display["confidence"].apply(format_float)
        df_rules_display["lift"] = df_rules_display["lift"].apply(format_float)
        st.dataframe(df_rules_display)

    with tabs[2]:
        st.header("📈 Visualizations")
        plot_bar_chart(freq_df)
        plot_pie_chart(freq_df)
        plot_bubble_chart(rules_df)
        plot_network_graph(rules_df)

    with tabs[3]:
        st.header("📄 Export Results")
        df_freq = freq_df.toPandas()
        df_rules = rules_df.toPandas()
        st.download_button("📅 Download Frequent Itemsets", df_freq.to_csv(index=False).encode(), file_name="frequent_itemsets.csv")
        st.download_button("📅 Download Association Rules", df_rules.to_csv(index=False).encode(), file_name="association_rules.csv")

    with tabs[4]:
        st.header("💡 Business Insights")
        df_rules = rules_df.toPandas()
        insights = df_rules[(df_rules["lift"] >= 1.2) & (df_rules["confidence"] >= 0.1)].head(5)
        if insights.empty:
            st.warning("No strong business insights found using default thresholds.")
            st.markdown("Fallback: Showing top available rules due to limited high-confidence insights.")
            insights = df_rules.sort_values(["lift", "confidence"], ascending=False).head(3)
        for _, row in insights.iterrows():
            st.markdown(f"""
                <div style='background-color: #2a2a2a; color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                <strong>Rule:</strong> If a customer buys <em>{row['antecedent']}</em>, they also tend to buy <em>{row['consequent']}</em>.<br>
                <strong>Lift:</strong> {format_float(row['lift'])} – measures strength of the rule.<br>
                <strong>Confidence:</strong> {format_float(row['confidence'])} – probability of consequent given antecedent.<br>
                <strong>Recommendation:</strong> Consider bundling or joint promotions.
                </div>
            """, unsafe_allow_html=True)

    with tabs[5]:
        st.header("📊 Dashboard Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("🧾 Total Transactions", df.count())
        col2.metric("📦 Frequent Itemsets", freq_df.count())
        col3.metric("🔗 Association Rules", rules_df.count())
        col1.metric("📈 Max Lift", format_float(rules_df.agg({"lift": "max"}).collect()[0][0]))
        col2.metric("📉 Avg Confidence", format_float(rules_df.agg({"confidence": "avg"}).collect()[0][0]))

    with tabs[6]:
        st.header("📄 Export Cleaned Transaction Data")
        df_cleaned_pd = df.limit(10000).toPandas()
        df_cleaned_pd["items"] = df_cleaned_pd["items"].apply(lambda x: ", ".join(x))
        csv_cleaned = df_cleaned_pd.to_csv(index=False).encode()
        st.download_button("⬇️ Download Cleaned Data (CSV)", data=csv_cleaned, file_name="cleaned_transactions.csv")
