# Shopping Mall Analytics Dashboard

A comprehensive analytics dashboard for shopping mall performance analysis, built with FastAPI and Streamlit.

**Live Demo:** [https://shopping-mall-analytics-dashboard.streamlit.app/](https://shopping-mall-analytics-dashboard.streamlit.app/)

## Overview

This application provides detailed analytics for shopping mall operations, including customer segmentation, sales trends, payment preferences, and campaign ROI simulations. The dashboard processes transaction data to deliver actionable insights for mall management and marketing teams.

## Features

### Analytics Modules

1. **Mall & Region Performance**
   - Total sales comparison across shopping malls
   - Regional sales distribution with donut chart visualization
   - Performance metrics by location

2. **Customer Analysis**
   - Repeat vs one-time customer segmentation
   - Customer value segmentation (High/Medium/Low value)
   - Top customers identification and analysis
   - RFM (Recency, Frequency, Monetary) segmentation

3. **Category Insights**
   - Sales performance by product category
   - Average order value analysis
   - Customer segment preferences by category

4. **Seasonality Analysis**
   - Daily sales trends
   - Monthly sales patterns
   - Quarterly year-over-year comparisons
   - Yearly sales evolution

5. **Payment Method Preference**
   - Distribution of payment methods
   - Transaction patterns by payment type

6. **Campaign Simulation**
   - ROI modeling for targeted discount campaigns
   - High-value customer targeting scenarios
   - Response rate and uplift analysis

## Technology Stack

### Backend
- **FastAPI** - RESTful API framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

### Frontend
- **Streamlit** - Interactive web application framework
- **BeautifulSoup4** - HTML parsing
- **Pillow** - Image processing

## Project Structure

```
mlops-deployed/
├── main.py                      # FastAPI backend with analytics endpoints
├── app_combined.py              # Streamlit frontend application
├── customer_shopping_data.csv   # Transaction data
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Installation & Setup

### Prerequisites
- Python 3.13+
- pip package manager

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/mlops-deployed.git
cd mlops-deployed
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app_combined.py
```

The FastAPI backend starts automatically in the background. Access the dashboard at `http://localhost:8501`

### Alternative: Separate Backend/Frontend

If you prefer to run them separately:

**Terminal 1 (Backend):**
```bash
uvicorn main:app --reload
```

**Terminal 2 (Frontend):**
```bash
streamlit run app.py
```

## Data Format

The application expects a CSV file with the following columns:
- `invoice_no` - Transaction identifier
- `customer_id` - Customer identifier
- `shopping_mall` - Mall name
- `category` - Product category
- `quantity` - Items purchased
- `price` - Unit price
- `payment_method` - Payment type
- `invoice_date` - Transaction date

## API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /` - Health check
- `GET /mall-region-performance` - Mall and region analytics
- `GET /repeat-vs-one-html` - Customer retention metrics
- `GET /category-insights` - Category performance charts
- `GET /top-customers` - Top customer analysis
- `GET /value-segmentation` - Customer value segments
- `GET /seasonality-analysis` - Time-based trends
- `GET /payment-method-preference` - Payment analysis
- `GET /rfm-analysis` - RFM segmentation
- `GET /campaign-simulation` - ROI simulation

API documentation available at `/docs` when running locally.

## Deployment

The application is deployed on Streamlit Cloud. To deploy your own instance:

1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and set main file to `app_combined.py`
5. Deploy

## Key Insights Provided

- **Customer Lifetime Value**: Identify and prioritize high-value customers
- **Sales Patterns**: Understand seasonal trends and peak periods
- **Category Performance**: Optimize inventory and marketing by category
- **Regional Differences**: Compare performance across locations
- **Campaign Effectiveness**: Simulate ROI before launching promotions
- **Payment Trends**: Understand customer payment preferences

## Use Cases

- **Mall Management**: Strategic planning and resource allocation
- **Marketing Teams**: Targeted campaign development
- **Sales Analysis**: Performance tracking and forecasting
- **Business Intelligence**: Data-driven decision making

## Future Enhancements

- Real-time data streaming
- Predictive analytics and forecasting
- Custom date range filtering
- Export functionality for reports
- Multi-language support
- Mobile app version

---
