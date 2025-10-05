import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime as dt
import matplotlib.dates as mdates
import base64
from io import BytesIO

# -----------------------------
# Helper: Convert matplotlib figure to base64 for HTML
# -----------------------------
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# -----------------------------
# Load data
# -----------------------------
st.set_page_config(layout="wide", page_title="Retail Analytics Dashboard")

df = pd.read_csv(
    r"C:\Users\sbert\Documents\MLOps\customer_shopping_data.csv",
    parse_dates=['invoice_date'], dayfirst=True, infer_datetime_format=True
)
df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
df["total_amount"] = df["price"] * df["quantity"]

# -----------------------------
# Region mapping
# -----------------------------
region_map = {
    'Kanyon': 'Europe',
    'Forum Istanbul': 'Europe',
    'Metrocity': 'Europe',
    'Metropol AVM': 'Asia',
    'Istinye Park': 'Europe',
    'Mall of Istanbul': 'Europe',
    'Emaar Square Mall': 'Asia',
    'Cevahir AVM': 'Europe',
    'Viaport Outlet': 'Asia',
    'Zorlu Center': 'Europe'
}
df["region"] = df["shopping_mall"].map(region_map)

# -----------------------------
# Tabs Layout
# -----------------------------
st.title("🛍️ Customer Shopping Insights Dashboard")
tabs = st.tabs([
    "Mall & Region Performance", "Top Customers", "Value Segmentation",
    "Seasonality Analysis", "Payment Method Preference", "RFM Analysis",
    "Category Insights", "Campaign Simulation"
])

# -----------------------------
# Tab 1 – Mall & Region Performance
# -----------------------------
with tabs[0]:
    st.header("🏬 Mall & Region Performance")
    mall_perf = df.groupby("shopping_mall")["total_amount"].sum().reset_index().sort_values(by="total_amount", ascending=False)
    region_perf = df.groupby("region")["total_amount"].sum().reset_index()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mall Performance")
        st.plotly_chart(px.bar(mall_perf, x="shopping_mall", y="total_amount", color="shopping_mall",
                               title="Total Sales by Mall", height=400))
    with col2:
        st.subheader("Region Performance")
        st.plotly_chart(px.pie(region_perf, names="region", values="total_amount", title="Sales by Region"))

# -----------------------------
# Tab 2 – Top Customers
# -----------------------------
with tabs[1]:
    st.header("👥 Top Customers Analysis")
    if 'net_sales' not in df.columns:
        df['net_sales'] = df['quantity'] * df['price']

    customer_summary = df.groupby('customer_id').agg(
        total_spent=('net_sales', 'sum'),
        total_qty=('quantity', 'sum'),
        num_invoices=('invoice_no', 'nunique')
    ).reset_index()
    customer_summary['avg_order_value'] = customer_summary['total_spent'] / customer_summary['num_invoices']

    col1, col2, col3 = st.columns(3)

    # Top 10 Customers by Spend
    top_10_customers = customer_summary.sort_values('total_spent', ascending=False).head(10)
    with col1:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=top_10_customers, x='customer_id', y='total_spent', palette='Blues_r')
        plt.xticks(rotation=45)
        plt.title("Top 10 Customers by Total Spend")
        plt.xlabel("Customer ID")
        plt.ylabel("Total Spend")
        st.pyplot(plt.gcf())
        plt.close()

    # Revenue Contribution by Segment (%)
    spend_threshold = customer_summary['total_spent'].quantile(0.90)
    customer_summary['segment'] = np.where(customer_summary['total_spent'] >= spend_threshold, 'Top 10%', 'Other')
    segment_contribution = customer_summary.groupby('segment')['total_spent'].sum()
    segment_contribution_pct = segment_contribution / segment_contribution.sum() * 100
    with col2:
        plt.figure(figsize=(6, 5))
        sns.barplot(x=segment_contribution_pct.index, y=segment_contribution_pct.values, palette="viridis")
        plt.title("Revenue Contribution by Segment (%)")
        plt.ylabel("Revenue Contribution %")
        plt.xlabel("Customer Segment")
        st.pyplot(plt.gcf())
        plt.close()

    # Cumulative Revenue Contribution by Customers
    customer_sorted = customer_summary.sort_values('total_spent', ascending=False).reset_index(drop=True)
    customer_sorted['cumulative_pct'] = customer_sorted['total_spent'].cumsum() / customer_sorted['total_spent'].sum() * 100
    with col3:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=customer_sorted, x=customer_sorted.index, y='cumulative_pct')
        plt.axhline(80, color='red', linestyle='--', label="80% threshold")
        plt.title("Cumulative Revenue Contribution by Customers")
        plt.xlabel("Customers (sorted by spend)")
        plt.ylabel("Cumulative % of Revenue")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

# -----------------------------
# Tab 3 – Value Segmentation
# -----------------------------
with tabs[2]:
    st.header("💎 Customer Value Segmentation")
    q30 = customer_summary['total_spent'].quantile(0.30)
    q80 = customer_summary['total_spent'].quantile(0.80)

    def classify_customer(spent):
        if spent >= q80:
            return "High-value"
        elif spent >= q30:
            return "Medium-value"
        else:
            return "Low-value"

    customer_summary['value_segment'] = customer_summary['total_spent'].apply(classify_customer)
    st.plotly_chart(px.histogram(customer_summary, x="value_segment", title="Customer Value Segments", color="value_segment"))

# -----------------------------
# Tab 4 – Seasonality Analysis
# -----------------------------
with tabs[3]:
    st.header("📅 Seasonality & Time Trends")
    df_copy = df.copy().dropna(subset=['invoice_date'])
    df_copy['net_sales'] = df_copy['price'] * df_copy['quantity']

    # Daily sales
    daily_sales = df_copy.groupby('invoice_date').agg(
        total_sales=('net_sales','sum'),
        total_qty=('quantity','sum'),
        num_invoices=('invoice_no','nunique')
    ).reset_index()

    # Daily plots
    fig, ax = plt.subplots(figsize=(12,5))
    sns.lineplot(data=daily_sales, x='invoice_date', y='total_sales', marker='o', ax=ax)
    ax.set_title("Daily Total Sales"); ax.set_xlabel("Date"); ax.set_ylabel("Total Sales")
    st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,5))
    sns.lineplot(data=daily_sales, x='invoice_date', y='num_invoices', marker='o', color='orange', ax=ax)
    ax.set_title("Daily Number of Invoices"); ax.set_xlabel("Date"); ax.set_ylabel("Invoices")
    st.pyplot(fig); plt.close(fig)

    # # Store-level daily 7-day MA
    # store_daily = df_copy.groupby(['shopping_mall','invoice_date']).agg(total_sales=('net_sales','sum')).reset_index()
    # store_daily['sales_ma7'] = store_daily.groupby('shopping_mall')['total_sales'].transform(lambda x: x.rolling(7,1).mean())
    # for mall in store_daily['shopping_mall'].unique():
    #     fig, ax = plt.subplots(figsize=(10,4))
    #     sns.lineplot(data=store_daily[store_daily['shopping_mall']==mall], x='invoice_date', y='sales_ma7', marker='o', ax=ax)
    #     ax.set_title(f"{mall} – 7-day MA Sales"); ax.set_xlabel("Date"); ax.set_ylabel("Sales")
    #     st.pyplot(fig); plt.close(fig)

    # Monthly sales
    df_copy['year_month'] = df_copy['invoice_date'].dt.to_period('M').astype(str)
    monthly_sales = df_copy.groupby('year_month')['net_sales'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12,5))
    sns.lineplot(data=monthly_sales, x='year_month', y='net_sales', marker='o', ax=ax)
    ax.set_title("Monthly Sales Trend"); ax.set_xlabel("Month"); ax.set_ylabel("Total Sales"); plt.xticks(rotation=45)
    st.pyplot(fig); plt.close(fig)

    # Quarterly sales
    df_copy['year_quarter'] = df_copy['invoice_date'].dt.to_period('Q').astype(str)
    quarterly_sales = df_copy.groupby('year_quarter')['net_sales'].sum().reset_index()
    quarterly_sales['year'] = quarterly_sales['year_quarter'].str[:4]
    quarterly_sales['quarter'] = quarterly_sales['year_quarter'].str[-2:]

    # Quarterly YOY Comparison
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=quarterly_sales, x='quarter', y='net_sales', hue='year', palette='viridis', ax=ax)
    ax.set_title("Quarterly Sales Comparison (YOY)"); ax.set_xlabel("Quarter"); ax.set_ylabel("Total Sales"); ax.legend(title="Year")
    st.pyplot(fig); plt.close(fig)

    # Quarterly Sales Trend
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=quarterly_sales, x='year_quarter', y='net_sales', palette='viridis', ax=ax)
    ax.set_title("Quarterly Sales Trend"); ax.set_xlabel("Quarter"); ax.set_ylabel("Total Sales")
    st.pyplot(fig); plt.close(fig)

    # Yearly sales
    df_copy['year'] = df_copy['invoice_date'].dt.year
    yearly_sales = df_copy.groupby('year')['net_sales'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=yearly_sales, x='year', y='net_sales', palette='magma', ax=ax)
    ax.set_title("Yearly Sales Comparison"); ax.set_xlabel("Year"); ax.set_ylabel("Total Sales")
    st.pyplot(fig); plt.close(fig)

# -----------------------------
# Tab 5 – Payment Method Preference
# -----------------------------
with tabs[4]:
    st.header("💳 Payment Method Preferences")
    if "payment_method" in df.columns:
        st.plotly_chart(px.pie(df, names="payment_method", values="total_amount", title="Payment Method Distribution"))
    else:
        st.warning("No 'payment_method' column found in dataset.")

# -----------------------------
# Tab 6 – RFM Analysis
# -----------------------------
with tabs[5]:
    st.header("📊 RFM Analysis")
    df_copy = df.copy().dropna(subset=['invoice_date'])
    df_copy['net_sales'] = df_copy['price'] * df_copy['quantity']
    reference_date = df_copy['invoice_date'].max() + dt.timedelta(days=1)

    recency_df = df_copy.groupby('customer_id')['invoice_date'].max().reset_index()
    recency_df['recency'] = (reference_date - recency_df['invoice_date']).dt.days
    frequency_df = df_copy.groupby('customer_id')['invoice_no'].nunique().reset_index().rename(columns={'invoice_no':'frequency'})
    monetary_df = df_copy.groupby('customer_id')['net_sales'].sum().reset_index().rename(columns={'net_sales':'monetary'})
    rfm = recency_df.merge(frequency_df,on='customer_id').merge(monetary_df,on='customer_id')

    rfm['R_score'] = pd.qcut(rfm['recency'],5,labels=[5,4,3,2,1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'),5,labels=[1,2,3,4,5]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['monetary'],5,labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_Score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

    def segment_customer(score):
        if score >= 12: return 'Champions'
        elif score >= 9: return 'Loyal'
        elif score >= 6: return 'At Risk'
        else: return 'Lost'
    rfm['Segment'] = rfm['RFM_Score'].apply(segment_customer)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index, palette="Set2", ax=ax)
    ax.set_title("Customer Segmentation Distribution (RFM)"); ax.set_ylabel("Number of Customers")
    st.pyplot(fig); plt.close(fig)

# -----------------------------
# Tab 7 – Category Insights
# -----------------------------
with tabs[6]:
    st.header("📦 Category Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(px.bar(df.groupby("category")["total_amount"].sum().reset_index(),
                               x="category", y="total_amount", title="Sales by Category"))
    with col2:
        st.plotly_chart(px.bar(df.groupby("category")["quantity"].sum().reset_index(),
                               x="category", y="quantity", title="Quantity Sold by Category"))
    with col3:
        avg_price = df.groupby("category")["total_amount"].mean().reset_index()
        st.plotly_chart(px.bar(avg_price, x="category", y="total_amount", title="Avg Spend per Category"))

# -----------------------------
# Tab 8 – Campaign Simulation
# -----------------------------
with tabs[7]:
    st.header("🎯 Campaign Simulation – 10% Discount Strategy")
    current_revenue = df["total_amount"].sum()
    discount = 0.10
    predicted_uplift = 0.15
    new_revenue = current_revenue * (1 - discount) * (1 + predicted_uplift)
    roi = ((new_revenue - current_revenue)/current_revenue)*100
    st.metric("Current Revenue", f"${current_revenue:,.0f}")
    st.metric("Projected Revenue (After 10% Discount)", f"${new_revenue:,.0f}")
    st.metric("Estimated ROI", f"{roi:.2f}%")
    st.plotly_chart(px.bar(x=["Current Revenue","Projected Revenue"], y=[current_revenue,new_revenue],
                           title="Revenue Comparison Before & After Campaign", labels={"x":"Scenario","y":"Revenue"}))


