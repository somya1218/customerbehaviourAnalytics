import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
try:
    from utils.data_generator import generate_customer_data, generate_realtime_data
    from utils.ai_insights import generate_insights, get_recommendations
    from utils.model_trainer import train_models, load_models
except ImportError:
    st.info("Utility modules will be loaded when available")

# Page configuration
st.set_page_config(
    page_title="Customer Behavior Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for interactive dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: fadeInDown 1s;
    }
    
    .dashboard-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .chart-preview {
        height: 200px;
        overflow: hidden;
        border-radius: 10px;
        position: relative;
    }
    
    .chart-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(102, 126, 234, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        transition: opacity 0.3s ease;
        cursor: pointer;
        border-radius: 10px;
    }
    
    .chart-overlay:hover {
        opacity: 1;
    }
    
    .expand-btn {
        background: white;
        color: #667eea;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .nav-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        margin: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .quick-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .alert-banner {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for dashboard navigation
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'dashboard'
if 'expanded_chart' not in st.session_state:
    st.session_state.expanded_chart = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_data
def generate_sample_data():
    """Generate comprehensive sample data"""
    np.random.seed(42)
    n_customers = 5000
    
    data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
        'age': np.random.normal(40, 15, n_customers).astype(int),
        'gender': np.random.choice(['M', 'F'], n_customers),
        'total_spent': np.random.lognormal(7, 1, n_customers),
        'purchase_frequency': np.random.choice(['Weekly', 'Monthly', 'Quarterly', 'Rarely'], n_customers),
        'days_since_last_purchase': np.random.exponential(30, n_customers).astype(int),
        'segment': np.random.choice(['High Value', 'Medium Value', 'Low Value', 'At Risk'], n_customers),
        'churn_probability': np.random.random(n_customers),
        'satisfaction_score': np.random.uniform(1, 5, n_customers),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_customers)
    }
    
    df = pd.DataFrame(data)
    df['churn'] = (df['churn_probability'] > 0.3).astype(int)
    df['age'] = np.clip(df['age'], 18, 80)
    df['total_spent'] = np.clip(df['total_spent'], 100, 50000)
    
    return df

def create_chart_preview(fig, chart_id, title):
    """Create a clickable chart preview"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show smaller version of chart
        fig.update_layout(height=200, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, key=f"preview_{chart_id}")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"🔍 Expand {title}", key=f"expand_{chart_id}", type="primary"):
            st.session_state.expanded_chart = chart_id

def show_expanded_chart(chart_id, data):
    """Show expanded version of selected chart"""
    st.markdown(f"### 📊 {chart_id.replace('_', ' ').title()}")
    
    if chart_id == "revenue_trend":
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        revenue_data = pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(50000, 10000, len(dates)).cumsum(),
            'target': np.linspace(50000, 20000000, len(dates))
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=revenue_data['date'], y=revenue_data['revenue'], 
                                mode='lines', name='Actual Revenue', line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=revenue_data['date'], y=revenue_data['target'], 
                                mode='lines', name='Target', line=dict(color='#764ba2', dash='dash', width=2)))
        
        fig.update_layout(height=600, title="Revenue Trend Analysis", 
                         xaxis_title="Date", yaxis_title="Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Month", "$1.2M", "12.5%")
        with col2:
            st.metric("YTD Growth", "45.2%", "8.3%")
        with col3:
            st.metric("Forecast Next Month", "$1.35M", "12.5%")
    
    elif chart_id == "customer_segments":
        segment_data = data['segment'].value_counts()
        
        # Main pie chart
        fig = px.pie(values=segment_data.values, names=segment_data.index, 
                    title='Customer Segments Distribution', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.markdown("#### Segment Analysis")
        for segment in segment_data.index:
            segment_customers = data[data['segment'] == segment]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"{segment} Count", len(segment_customers))
            with col2:
                st.metric("Avg Spend", f"${segment_customers['total_spent'].mean():,.0f}")
            with col3:
                st.metric("Avg Age", f"{segment_customers['age'].mean():.1f}")
            with col4:
                st.metric("Satisfaction", f"{segment_customers['satisfaction_score'].mean():.2f}/5")
    
    elif chart_id == "churn_analysis":
        churn_data = data.groupby('segment')['churn'].mean() * 100
        
        fig = px.bar(x=churn_data.index, y=churn_data.values, 
                    title='Churn Rate by Segment', height=600,
                    color=churn_data.values, color_continuous_scale='Reds')
        fig.update_layout(xaxis_title="Customer Segment", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis
        high_risk = data[data['churn_probability'] > 0.7]
        st.markdown(f"#### 🚨 High Risk Customers: {len(high_risk)}")
        st.dataframe(high_risk[['customer_id', 'segment', 'churn_probability', 'days_since_last_purchase']].head(10))
    
    # Back button
    if st.button("⬅️ Back to Dashboard", type="secondary"):
        st.session_state.expanded_chart = None
        st.rerun()

def main():
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading customer data..."):
            st.session_state.customer_data = generate_sample_data()
            st.session_state.data_loaded = True
    
    data = st.session_state.customer_data
    
    # Header
    st.markdown('<h1 class="main-header">📊 Interactive Customer Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Navigation")
        
        # Quick navigation buttons
        nav_options = {
            "🏠 Main Dashboard": "dashboard",
            "📊 Chart Explorer": "charts",
            "👥 Segment Deep Dive": "segments", 
            "🔮 Predictive Analytics": "predictions",
            "💡 AI Insights": "insights",
            "⚙️ Settings": "settings"
        }
        
        for label, view in nav_options.items():
            if st.button(label, key=f"nav_{view}", type="primary" if st.session_state.current_view == view else "secondary"):
                st.session_state.current_view = view
                st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Customers", f"{len(data):,}")
        st.metric("Avg Revenue", f"${data['total_spent'].mean():,.0f}")
        st.metric("Churn Rate", f"{data['churn'].mean()*100:.1f}%")
        st.metric("Satisfaction", f"{data['satisfaction_score'].mean():.2f}/5")
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        selected_segments = st.multiselect(
            "Customer Segments",
            data['segment'].unique(),
            default=data['segment'].unique()
        )
        
        age_range = st.slider("Age Range", 18, 80, (18, 80))
        
        # Apply filters
        filtered_data = data[
            (data['segment'].isin(selected_segments)) &
            (data['age'].between(age_range[0], age_range[1]))
        ]
        
        st.success(f"📊 {len(filtered_data):,} customers selected")
    
    # Show expanded chart if selected
    if st.session_state.expanded_chart:
        show_expanded_chart(st.session_state.expanded_chart, filtered_data)
        return
    
    # Main dashboard views
    if st.session_state.current_view == "dashboard":
        # Alert banner for high-risk customers
        high_risk_count = len(filtered_data[filtered_data['churn_probability'] > 0.7])
        if high_risk_count > 0:
            st.markdown(f"""
                <div class="alert-banner">
                    🚨 <strong>Alert:</strong> {high_risk_count} customers are at high risk of churning!
                    <button onclick="document.getElementById('churn_analysis').scrollIntoView();">View Details →</button>
                </div>
            """, unsafe_allow_html=True)
        
        # Key Performance Metrics
        st.markdown("## 🎯 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics_data = [
            ("Total Revenue", f"${filtered_data['total_spent'].sum():,.0f}", "12.5%"),
            ("Active Customers", f"{len(filtered_data):,}", "5.2%"),
            ("Avg Order Value", f"${filtered_data['total_spent'].mean():,.0f}", "8.1%"),
            ("Customer Lifetime Value", "$2,350", "15.3%"),
            ("Net Promoter Score", "72", "3.2%")
        ]
        
        for i, (label, value, change) in enumerate(metrics_data):
            with [col1, col2, col3, col4, col5][i]:
                st.markdown(f"""
                    <div class="metric-card" onclick="alert('Detailed {label} analysis coming soon!')">
                        <h3>{value}</h3>
                        <p>{label}</p>
                        <small>↗️ {change}</small>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interactive Chart Grid
        st.markdown("## 📊 Interactive Analytics Grid")
        st.markdown("*Click on any chart to explore in detail*")
        
        # Row 1: Revenue and Segments
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Revenue Trend")
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
            revenue_trend = pd.DataFrame({
                'date': dates,
                'revenue': np.random.normal(200000, 50000, len(dates)).cumsum()
            })
            fig_revenue = px.line(revenue_trend, x='date', y='revenue', 
                                title='Weekly Revenue Trend')
            create_chart_preview(fig_revenue, "revenue_trend", "Revenue Analysis")
        
        with col2:
            st.markdown("#### 👥 Customer Segments")
            segment_data = filtered_data['segment'].value_counts()
            fig_segments = px.pie(values=segment_data.values, names=segment_data.index,
                                title='Customer Distribution')
            create_chart_preview(fig_segments, "customer_segments", "Segment Analysis")
        
        # Row 2: Churn and Geography
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚠️ Churn Analysis")
            churn_data = filtered_data.groupby('segment')['churn'].mean() * 100
            fig_churn = px.bar(x=churn_data.index, y=churn_data.values,
                             title='Churn Rate by Segment')
            create_chart_preview(fig_churn, "churn_analysis", "Churn Details")
        
        with col2:
            st.markdown("#### 🌍 Regional Performance")
            region_data = filtered_data.groupby('region')['total_spent'].sum()
            fig_region = px.bar(x=region_data.index, y=region_data.values,
                              title='Revenue by Region')
            create_chart_preview(fig_region, "regional_performance", "Regional Analysis")
    
    elif st.session_state.current_view == "charts":
        st.markdown("## 📊 Chart Explorer")
        st.markdown("Explore all available visualizations")
        
        # Chart categories
        chart_categories = {
            "📈 Financial Charts": ["revenue_trend", "profit_margin", "cost_analysis"],
            "👥 Customer Analytics": ["customer_segments", "satisfaction_analysis", "loyalty_metrics"],
            "🔮 Predictive Models": ["churn_analysis", "demand_forecast", "lifetime_value"],
            "🌍 Geographic Analysis": ["regional_performance", "market_penetration", "location_trends"]
        }
        
        for category, charts in chart_categories.items():
            with st.expander(category):
                cols = st.columns(len(charts))
                for i, chart_id in enumerate(charts):
                    with cols[i]:
                        if st.button(f"📊 {chart_id.replace('_', ' ').title()}", 
                                   key=f"chart_btn_{chart_id}"):
                            st.session_state.expanded_chart = chart_id
                            st.rerun()
    
    elif st.session_state.current_view == "segments":
        st.markdown("## 👥 Customer Segment Deep Dive")
        
        # Segment selector
        selected_segment = st.selectbox(
            "Choose segment to analyze",
            filtered_data['segment'].unique()
        )
        
        segment_data = filtered_data[filtered_data['segment'] == selected_segment]
        
        # Segment metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Customer Count", len(segment_data))
        with col2:
            st.metric("Avg Revenue", f"${segment_data['total_spent'].mean():,.0f}")
        with col3:
            st.metric("Churn Rate", f"{segment_data['churn'].mean()*100:.1f}%")
        with col4:
            st.metric("Satisfaction", f"{segment_data['satisfaction_score'].mean():.2f}/5")
        
        # Detailed analysis
        tab1, tab2, tab3 = st.tabs(["📊 Demographics", "💰 Purchase Behavior", "🎯 Recommendations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig_age = px.histogram(segment_data, x='age', title='Age Distribution')
                st.plotly_chart(fig_age, use_container_width=True)
            with col2:
                fig_gender = px.pie(segment_data, names='gender', title='Gender Distribution')
                st.plotly_chart(fig_gender, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig_freq = px.histogram(segment_data, x='purchase_frequency', 
                                      title='Purchase Frequency')
                st.plotly_chart(fig_freq, use_container_width=True)
            with col2:
                fig_spend = px.histogram(segment_data, x='total_spent', 
                                       title='Spending Distribution')
                st.plotly_chart(fig_spend, use_container_width=True)
        
        with tab3:
            st.markdown("### 💡 AI-Powered Recommendations")
            recommendations = [
                "🎯 Target with personalized email campaigns",
                "💰 Offer loyalty program enrollment", 
                "📱 Increase mobile app engagement",
                "🎁 Provide exclusive member benefits",
                "📊 Monitor satisfaction scores closely"
            ]
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    # Real-time updates simulation
    if st.session_state.get('live_mode', False):
        st.markdown("### ⚡ Live Data Stream")
        placeholder = st.empty()
        
        # Simulate real-time updates
        import time
        for i in range(5):
            with placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Live Orders", np.random.randint(1000, 2000))
                with col2:
                    st.metric("Active Users", np.random.randint(5000, 8000))
                with col3:
                    st.metric("Revenue Today", f"${np.random.randint(50000, 100000):,}")
            time.sleep(1)

if __name__ == "__main__":
    main()