# Shopping Mall Analytics Dashboard - Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Backend Documentation](#backend-documentation)
3. [Frontend Documentation](#frontend-documentation)
4. [Data Model](#data-model)
5. [API Reference](#api-reference)
6. [Deployment Guide](#deployment-guide)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit Frontend                │
│                  (app_combined.py)                  │
│  - User Interface                                   │
│  - Chart Rendering                                  │
│  - Navigation                                       │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP Requests
                   ↓
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend (main.py)              │
│  - Data Processing                                  │
│  - Analytics Computation                            │
│  - Chart Generation                                 │
└──────────────────┬──────────────────────────────────┘
                   │ Read
                   ↓
┌─────────────────────────────────────────────────────┐
│           CSV Data Store                            │
│     (customer_shopping_data.csv)                    │
└─────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend Framework:**
- FastAPI 0.115.0 - Modern, fast web framework for building APIs
- Uvicorn 0.30.0 - ASGI server implementation

**Data Processing:**
- Pandas 2.2.3 - Data manipulation and analysis
- NumPy 1.26.4 - Numerical computing

**Visualization:**
- Matplotlib 3.9.0 - Comprehensive 2D plotting
- Seaborn 0.13.2 - Statistical data visualization

**Frontend:**
- Streamlit 1.39.0 - Web app framework
- Requests 2.32.3 - HTTP library
- BeautifulSoup4 4.12.3 - HTML parsing
- Pillow 10.4.0 - Image processing

---

## Backend Documentation

### main.py Structure

#### 1. Data Loading and Preprocessing

```python
df = pd.read_csv("customer_shopping_data.csv", parse_dates=['invoice_date'])
```

**Key Operations:**
- Parses dates automatically for time-series analysis
- Loads entire dataset into memory
- Creates derived column `net_sales = price × quantity`

#### 2. Region Mapping

```python
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
```

**Purpose:** Groups malls by geographic region for regional analysis.

#### 3. Utility Functions

**`fig_to_base64(fig)`**
- Converts matplotlib figures to base64-encoded PNG
- Enables embedding charts in HTML responses
- Automatically closes figures to prevent memory leaks

```python
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64
```

**`compute_rfm(df)`**
- Shared function for RFM analysis
- Calculates Recency, Frequency, Monetary scores
- Segments customers into Champions, Loyal, At Risk, Lost

### Endpoint Implementations

#### 1. Mall & Region Performance (`/mall-region-performance`)

**Data Aggregation:**
```python
mall_perf = df.groupby('shopping_mall').agg(
    total_sales=('net_sales', 'sum'),
    total_qty=('quantity', 'sum'),
    unique_customers=('customer_id', 'nunique'),
    num_invoices=('invoice_no', 'nunique')
).reset_index()
```

**Visualizations:**
- Bar chart: Total sales by mall (sorted descending)
- Donut chart: Regional sales distribution

**Response:** HTML with embedded base64 images

#### 2. Repeat vs One-time Customers (`/repeat-vs-one-html`)

**Logic:**
```python
customer_orders = df.groupby('customer_id')['invoice_no'].nunique()
customer_type = np.where(num_invoices > 1, 'Repeat', 'One-time')
```

**Output:** HTML table with total sales by customer type

#### 3. Category Insights (`/category-insights`)

**Three Visualizations:**

1. **Total Sales by Category** (Bar chart)
2. **Average Order Value by Category** (Bar chart)
3. **Category Preferences by Customer Segment** (Heatmap)

**Key Calculation:**
```python
cat_segment_perf['pct_within_category'] = (
    cat_segment_perf.groupby('category')['total_sales']
    .transform(lambda x: x / x.sum() * 100)
)
```

#### 4. Top Customers Analysis (`/top-customers`)

**Components:**

1. **Top 10 Customers** - Bar chart of highest spenders
2. **Segment Contribution** - Top 10% vs Others revenue split
3. **Cumulative Revenue** - Pareto chart showing revenue concentration

**Segmentation Logic:**
```python
spend_threshold = customer_summary['total_spent'].quantile(0.90)
segment = np.where(total_spent >= threshold, 'Top 10%', 'Other')
```

#### 5. Value Segmentation (`/value-segmentation`)

**Quantile-based Segmentation:**
```python
q80 = customer_summary['total_spent'].quantile(0.80)
q30 = customer_summary['total_spent'].quantile(0.30)

if spent >= q80: 'High-value'
elif spent >= q30: 'Medium-value'
else: 'Low-value'
```

#### 6. Seasonality Analysis (`/seasonality-analysis`)

**Time Aggregations:**
- **Daily:** `groupby('invoice_date')`
- **Monthly:** `dt.to_period('M')`
- **Quarterly:** `dt.to_period('Q')`
- **Yearly:** `dt.year`

**7 Visualizations:**
1. Daily total sales line chart
2. Daily invoice count line chart
3. Monthly sales trend
4. Quarterly YOY comparison
5. Quarterly sales trend
6. Yearly sales comparison

#### 7. Payment Method Preference (`/payment-method-preference`)

**Calculation:**
```python
payment_dist = df['payment_method'].value_counts(normalize=True) * 100
```

**Output:** Bar chart showing percentage distribution

#### 8. RFM Analysis (`/rfm-analysis`)

**RFM Calculation:**

**Recency:**
```python
recency = (reference_date - last_purchase_date).days
```

**Frequency:**
```python
frequency = df.groupby('customer_id')['invoice_no'].nunique()
```

**Monetary:**
```python
monetary = df.groupby('customer_id')['net_sales'].sum()
```

**Scoring:**
- Each metric divided into 5 quintiles
- Recency: Lower is better (5 for recent, 1 for old)
- Frequency: Higher is better (5 for frequent, 1 for infrequent)
- Monetary: Higher is better (5 for high spend, 1 for low spend)

**Segmentation Rules:**
```python
RFM_Score = R_score + F_score + M_score

if score >= 12: 'Champions'
elif score >= 9: 'Loyal'
elif score >= 6: 'At Risk'
else: 'Lost'
```

#### 9. Campaign Simulation (`/campaign-simulation`)

**Model Parameters:**
- Target: Top 10% customers by spend
- Discount: 10%
- Variables: Response rate (5%-50%), Uplift (5%-50%)

**ROI Formula:**
```python
responding_customers = top_10_count × response_rate
additional_revenue = responding_customers × avg_spend × uplift
discount_cost = responding_customers × new_spend × discount_rate
net_gain = additional_revenue - discount_cost
ROI = (net_gain / discount_cost) × 100
```

**Output:** Heatmap showing ROI for different scenarios

---

## Frontend Documentation

### app_combined.py Structure

#### 1. FastAPI Integration

```python
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="error")

if 'fastapi_started' not in st.session_state:
    st.session_state.fastapi_started = True
    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()
    time.sleep(3)
```

**Purpose:** 
- Runs FastAPI backend in background thread
- Uses session state to prevent multiple instances
- 3-second delay ensures backend is ready

#### 2. Core Functions

**`fetch_html_content(endpoint)`**
- Makes GET request to FastAPI endpoint
- 30-second timeout for long-running analytics
- Returns HTML content or None on error

**`extract_images_from_html(html_content)`**
- Parses HTML using BeautifulSoup
- Extracts base64-encoded images from `<img>` tags
- Returns list of base64 strings

**`display_base64_image(base64_str)`**
- Decodes base64 to binary
- Opens with PIL/Pillow
- Displays using `st.image()`

#### 3. Navigation Structure

**Sidebar Radio Buttons:**
- Home
- Mall & Region Performance
- Repeat vs One-time Customers
- Category Insights
- Top Customers Analysis
- Value Segmentation
- Seasonality Analysis
- Payment Method Preference
- RFM Analysis
- Campaign Simulation

**Page Rendering Pattern:**
```python
if page == "Page Name":
    st.header("Page Title")
    html_content = fetch_html_content("/endpoint")
    if html_content:
        images = extract_images_from_html(html_content)
        for idx, img in enumerate(images):
            st.subheader("Chart Title")
            display_base64_image(img)
```

---

## Data Model

### Input Data Schema

**File:** `customer_shopping_data.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| invoice_no | string | Unique transaction identifier | INV-001234 |
| customer_id | string | Unique customer identifier | CUST-5678 |
| gender | string | Customer gender | Male/Female |
| age | integer | Customer age | 35 |
| category | string | Product category | Clothing |
| quantity | integer | Number of items | 2 |
| price | float | Unit price (currency) | 49.99 |
| payment_method | string | Payment type | Credit Card |
| invoice_date | datetime | Transaction timestamp | 2023-01-15 |
| shopping_mall | string | Mall name | Mall of Istanbul |

### Derived Columns

**net_sales:**
```python
net_sales = quantity × price
```
Total revenue per transaction line item.

**region:**
```python
region = region_map[shopping_mall]
```
Geographic region (Europe/Asia) mapped from mall name.

**year_month:**
```python
year_month = invoice_date.dt.to_period('M')
```
Period representation for monthly aggregation.

**year_quarter:**
```python
year_quarter = invoice_date.dt.to_period('Q')
```
Period representation for quarterly aggregation.

---

## API Reference

### Base URL
- **Local:** `http://localhost:8000`
- **Production:** Embedded in Streamlit app

### Endpoints

#### GET `/`
**Description:** Health check endpoint

**Response:**
```json
{
  "message": "Welcome to the Shopping Mall Analytics API! Visit /docs for API docs."
}
```

#### GET `/mall-region-performance`
**Description:** Mall and regional performance analysis

**Response:** HTML with embedded charts
- Bar chart: Sales by mall
- Donut chart: Sales by region

**Performance:** ~2-3 seconds

#### GET `/repeat-vs-one-html`
**Description:** Customer retention analysis

**Response:** HTML table

**Metrics:**
- Customer Type (Repeat/One-time)
- Total Sales

#### GET `/category-insights`
**Description:** Product category performance analysis

**Response:** HTML with 3 charts
1. Total sales by category
2. Average order value by category
3. Category preferences heatmap by customer segment

**Performance:** ~3-4 seconds

#### GET `/top-customers`
**Description:** High-value customer identification

**Response:** HTML with 3 charts
1. Top 10 customers bar chart
2. Segment contribution (Top 10% vs Others)
3. Cumulative revenue Pareto chart

#### GET `/value-segmentation`
**Description:** Customer segmentation by value

**Response:** HTML with bar chart

**Segments:**
- High-value (top 20%)
- Medium-value (30th-80th percentile)
- Low-value (bottom 30%)

#### GET `/seasonality-analysis`
**Description:** Time-based sales trends

**Response:** HTML with 6 charts
1. Daily total sales
2. Daily invoice count
3. Monthly sales trend
4. Quarterly YOY comparison
5. Quarterly sales trend
6. Yearly comparison

**Performance:** ~4-6 seconds (most intensive)

#### GET `/payment-method-preference`
**Description:** Payment method distribution

**Response:** HTML with bar chart

**Metrics:** Percentage of transactions by payment type

#### GET `/rfm-analysis`
**Description:** RFM customer segmentation

**Response:** HTML with bar chart

**Segments:**
- Champions (RFM Score ≥ 12)
- Loyal (9-11)
- At Risk (6-8)
- Lost (< 6)

#### GET `/campaign-simulation`
**Description:** ROI simulation for discount campaigns

**Response:** HTML with heatmap and metrics

**Model:**
- Target: Top 10% customers
- Discount: 10%
- Variables: Response rate, Uplift

**Output:** ROI percentage for various scenarios

---

## Deployment Guide

### Prerequisites
- GitHub account
- Streamlit Cloud account (free)
- Python 3.13+ compatible code

### Step-by-Step Deployment

#### 1. Prepare Repository

**Required Files:**
```
repo/
├── main.py
├── app_combined.py
├── customer_shopping_data.csv
├── requirements.txt
└── README.md
```

**requirements.txt:**
```txt
fastapi==0.115.0
uvicorn==0.30.0
pandas==2.2.3
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
streamlit==1.39.0
requests==2.32.3
beautifulsoup4==4.12.3
pillow==10.4.0
```

#### 2. GitHub Setup

1. Create new repository on GitHub
2. Make it **Public** (required for Streamlit Cloud free tier)
3. Upload all files
4. Verify CSV file is included (important!)

#### 3. Streamlit Cloud Deployment

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Repository: `your-username/your-repo`
   - Branch: `main`
   - Main file path: `app_combined.py`
5. Click "Deploy"

#### 4. Monitor Deployment

**Deployment Logs:**
- Package installation: ~2-3 minutes
- First startup: ~30 seconds
- Subsequent cold starts: ~10 seconds

**Common Issues:**
- Package version conflicts → Update requirements.txt
- CSV not found → Check file path is relative
- Import errors → Verify all files pushed to GitHub

#### 5. Access Your App

**URL Format:**
```
https://[app-name]-[random-string].streamlit.app/
```

**Example:**
```
https://shopping-mall-analytics-dashboard.streamlit.app/
```

### Environment Configuration

**Streamlit Secrets (Optional):**

For sensitive data, use secrets management:

1. In Streamlit Cloud dashboard, go to app settings
2. Click "Secrets"
3. Add configuration:

```toml
[data]
csv_url = "https://your-storage.com/data.csv"

[api]
api_key = "your-secret-key"
```

Access in code:
```python
import streamlit as st
csv_url = st.secrets["data"]["csv_url"]
```

---

## Troubleshooting

### Common Issues

#### 1. "Module not found" Error

**Problem:** Missing package in requirements.txt

**Solution:**
```bash
pip freeze > requirements.txt
# Review and clean up unnecessary packages
```

#### 2. FastAPI Not Starting

**Problem:** Port conflict or threading issue

**Solution:**
- Check if port 8000 is available
- Verify session state initialization
- Increase startup delay: `time.sleep(5)`

#### 3. Charts Not Displaying

**Problem:** base64 encoding/decoding failure

**Solution:**
- Check matplotlib backend: `matplotlib.use('Agg')`
- Verify image format: `format="png"`
- Test base64 string validity

#### 4. Slow Performance

**Problem:** Large dataset or inefficient queries

**Solutions:**
- Implement caching: `@st.cache_data`
- Optimize pandas operations
- Consider data sampling for development

#### 5. Memory Issues

**Problem:** Multiple matplotlib figures not closed

**Solution:**
Always close figures after saving:
```python
plt.close(fig)
# or
plt.close('all')
```

#### 6. CSV File Not Found

**Problem:** Absolute path in code

**Wrong:**
```python
df = pd.read_csv(r"C:\Users\...\data.csv")
```

**Correct:**
```python
df = pd.read_csv("customer_shopping_data.csv")
```

### Debugging Tips

**Enable Verbose Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Streamlit Session State:**
```python
st.write(st.session_state)
```

**Test Endpoints Directly:**
```bash
curl http://localhost:8000/mall-region-performance
```

**Monitor Resource Usage:**
```python
import psutil
memory = psutil.Process().memory_info().rss / 1024 / 1024
st.write(f"Memory: {memory:.2f} MB")
```

---

## Performance Optimization

### Backend Optimizations

#### 1. Data Loading

**Current:**
```python
df = pd.read_csv("customer_shopping_data.csv")
```

**Optimized:**
```python
# Specify dtypes to reduce memory
dtypes = {
    'customer_id': 'category',
    'shopping_mall': 'category',
    'category': 'category',
    'payment_method': 'category'
}
df = pd.read_csv("customer_shopping_data.csv", dtype=dtypes)
```

**Memory Reduction:** ~30-50%

#### 2. Caching Aggregations

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_mall_performance():
    return df.groupby('shopping_mall').agg(...)
```

#### 3. Vectorized Operations

**Instead of:**
```python
for index, row in df.iterrows():
    # Process each row
```

**Use:**
```python
df['result'] = df.apply(lambda x: process(x), axis=1)
# or better
df['result'] = vectorized_operation(df['column'])
```

### Frontend Optimizations

#### 1. Streamlit Caching

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_html_content(endpoint):
    response = requests.get(f"{API_BASE_URL}{endpoint}")
    return response.text
```

#### 2. Lazy Loading

```python
if page == "Seasonality Analysis":
    with st.spinner("Loading analysis..."):
        html_content = fetch_html_content("/seasonality-analysis")
```

#### 3. Image Optimization

```python
def display_base64_image(base64_str, width=None):
    img = Image.open(BytesIO(img_data))
    # Resize if needed
    if width and img.width > width:
        ratio = width / img.width
        new_size = (width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    st.image(img, use_container_width=True)
```

### Database Considerations

For production with large datasets, consider:

**PostgreSQL:**
```python
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@host/db')
df = pd.read_sql_query("SELECT * FROM transactions", engine)
```

**Benefits:**
- Indexed queries
- Incremental loading
- Better memory management
- Concurrent access

### Monitoring

**Add Performance Metrics:**
```python
import time

start_time = time.time()
# Expensive operation
duration = time.time() - start_time
st.sidebar.metric("Load Time", f"{duration:.2f}s")
```

---

## Appendix

### A. Color Schemes

**Chart Palettes:**
- Sequential: `'Blues_d'`, `'viridis'`, `'magma'`
- Diverging: `'RdYlGn'`, `'coolwarm'`
- Qualitative: `'Set2'`, `'tab10'`

### B. Date Formats

```python
# Display formats
df['invoice_date'].dt.strftime('%Y-%m-%d')
df['invoice_date'].dt.strftime('%B %d, %Y')  # January 15, 2023
```

### C. Number Formatting

```python
# Currency
f"${value:,.2f}"  # $1,234.56

# Percentage
f"{value:.1f}%"   # 45.2%

# Thousands
f"{value:,}"      # 1,234,567
```

### D. Error Handling Best Practices

```python
try:
    result = risky_operation()
except ValueError as e:
    st.error(f"Invalid value: {e}")
except Exception as e:
    st.error(f"Unexpected error: {e}")
    logging.exception("Detailed error")
finally:
    # Cleanup
    plt.close('all')
```

---

**Documentation Version:** 1.0  
**Last Updated:** October 2025  
**Maintained By:** Project Team
