import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_customer_data(n_customers=5000, seed=42):
    """
    Generate comprehensive customer behavior dataset
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Basic customer information
    customers = {
        'customer_id': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
        'age': np.random.normal(40, 15, n_customers).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_customers, p=[0.48, 0.52]),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_customers),
        'city_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n_customers, p=[0.3, 0.4, 0.3]),
        'signup_date': [
            datetime.now() - timedelta(days=random.randint(30, 1095)) 
            for _ in range(n_customers)
        ]
    }
    
    # Purchase behavior
    customers['total_spent'] = np.random.lognormal(7, 1.2, n_customers)
    customers['total_orders'] = np.random.poisson(8, n_customers) + 1
    customers['avg_order_value'] = customers['total_spent'] / customers['total_orders']
    
    # Purchase frequency
    customers['purchase_frequency'] = np.random.choice(
        ['Weekly', 'Bi-weekly', 'Monthly', 'Quarterly', 'Rarely'], 
        n_customers, 
        p=[0.15, 0.20, 0.35, 0.20, 0.10]
    )
    
    # Time-based features
    customers['days_since_last_purchase'] = np.random.exponential(45, n_customers).astype(int)
    customers['recency_score'] = np.where(
        customers['days_since_last_purchase'] <= 30, 5,
        np.where(customers['days_since_last_purchase'] <= 90, 4,
        np.where(customers['days_since_last_purchase'] <= 180, 3,
        np.where(customers['days_since_last_purchase'] <= 365, 2, 1)))
    )
    
    # Product preferences
    customers['preferred_category'] = np.random.choice([
        'Electronics', 'Clothing', 'Books', 'Home & Garden', 
        'Sports', 'Beauty', 'Automotive', 'Food'
    ], n_customers)
    
    customers['brand_loyalty_score'] = np.random.uniform(1, 5, n_customers)
    
    # Customer satisfaction and engagement
    customers['satisfaction_score'] = np.random.normal(3.5, 1.2, n_customers)
    customers['satisfaction_score'] = np.clip(customers['satisfaction_score'], 1, 5)
    
    customers['website_visits'] = np.random.poisson(12, n_customers)
    customers['mobile_app_usage'] = np.random.choice([0, 1], n_customers, p=[0.4, 0.6])
    customers['newsletter_subscribed'] = np.random.choice([0, 1], n_customers, p=[0.3, 0.7])
    
    # Support interactions
    customers['support_tickets'] = np.random.poisson(1.5, n_customers)
    customers['support_satisfaction'] = np.random.normal(3.8, 1.1, n_customers)
    customers['support_satisfaction'] = np.clip(customers['support_satisfaction'], 1, 5)
    
    # Convert to DataFrame
    df = pd.DataFrame(customers)
    
    # Clean age data
    df['age'] = np.clip(df['age'], 18, 85)
    
    # Calculate customer lifetime value
    df['customer_lifetime_value'] = df['total_spent'] * (df['brand_loyalty_score'] / 5) * 1.5
    
    # Calculate RFM scores
    df['frequency_score'] = pd.qcut(df['total_orders'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['monetary_score'] = pd.qcut(df['total_spent'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['recency_score'] * 100 + df['frequency_score'] * 10 + df['monetary_score']
    
    # Customer segmentation based on RFM
    def assign_segment(row):
        if row['rfm_score'] >= 444:
            return 'Champions'
        elif row['rfm_score'] >= 334:
            return 'Loyal Customers'
        elif row['rfm_score'] >= 324:
            return 'Potential Loyalists'
        elif row['rfm_score'] >= 314:
            return 'New Customers'
        elif row['rfm_score'] >= 244:
            return 'Promising'
        elif row['rfm_score'] >= 234:
            return 'Need Attention'
        elif row['rfm_score'] >= 224:
            return 'About to Sleep'
        elif row['rfm_score'] >= 214:
            return 'At Risk'
        elif row['rfm_score'] >= 144:
            return 'Cannot Lose Them'
        elif row['rfm_score'] >= 134:
            return 'Hibernating'
        else:
            return 'Lost'
    
    df['detailed_segment'] = df.apply(assign_segment, axis=1)
    
    # Simplified segments for main dashboard
    segment_mapping = {
        'Champions': 'High Value',
        'Loyal Customers': 'High Value',
        'Potential Loyalists': 'Medium Value',
        'New Customers': 'Medium Value',
        'Promising': 'Medium Value',
        'Need Attention': 'Low Value',
        'About to Sleep': 'At Risk',
        'At Risk': 'At Risk',
        'Cannot Lose Them': 'At Risk',
        'Hibernating': 'Low Value',
        'Lost': 'At Risk'
    }
    
    df['segment'] = df['detailed_segment'].map(segment_mapping)
    
    # Churn prediction features
    df['churn_probability'] = (
        (5 - df['recency_score']) * 0.3 +
        (5 - df['frequency_score']) * 0.25 +
        (5 - df['monetary_score']) * 0.2 +
        (5 - df['satisfaction_score']) * 0.15 +
        (df['support_tickets'] / 10) * 0.1
    ) / 5
    
    df['churn_probability'] = np.clip(df['churn_probability'], 0, 1)
    df['churn'] = (df['churn_probability'] > 0.6).astype(int)
    
    # Add some seasonal trends
    df['signup_month'] = df['signup_date'].dt.month
    df['signup_quarter'] = df['signup_date'].dt.quarter
    
    # Payment preferences
    df['preferred_payment'] = np.random.choice([
        'Credit Card', 'Debit Card', 'Digital Wallet', 
        'Bank Transfer', 'Cash on Delivery'
    ], n_customers, p=[0.35, 0.25, 0.2, 0.1, 0.1])
    
    # Device preferences
    df['preferred_device'] = np.random.choice([
        'Desktop', 'Mobile', 'Tablet'
    ], n_customers, p=[0.3, 0.6, 0.1])
    
    return df

def generate_realtime_data(base_data, n_new_records=100):
    """
    Generate real-time customer behavior data
    """
    # Simulate new customer acquisitions
    new_customers = generate_customer_data(n_new_records, seed=None)
    
    # Simulate existing customer activities
    existing_sample = base_data.sample(n=min(500, len(base_data)))
    
    # Update their recent activities
    existing_sample = existing_sample.copy()
    existing_sample['days_since_last_purchase'] = np.random.exponential(10, len(existing_sample))
    existing_sample['website_visits'] += np.random.poisson(2, len(existing_sample))
    
    # Some customers made new purchases
    purchase_mask = np.random.random(len(existing_sample)) < 0.3
    existing_sample.loc[purchase_mask, 'total_orders'] += 1
    existing_sample.loc[purchase_mask, 'total_spent'] += np.random.lognormal(5, 0.8, purchase_mask.sum())
    
    return {
        'new_customers': new_customers,
        'updated_customers': existing_sample,
        'timestamp': datetime.now()
    }

def generate_transaction_data(customer_data, n_transactions=10000):
    """
    Generate detailed transaction data
    """
    np.random.seed(42)
    
    transactions = []
    
    for _ in range(n_transactions):
        customer = customer_data.sample(1).iloc[0]
        
        transaction = {
            'transaction_id': f'TXN_{random.randint(100000, 999999)}',
            'customer_id': customer['customer_id'],
            'transaction_date': datetime.now() - timedelta(
                days=random.randint(1, 365)
            ),
            'product_category': customer['preferred_category'],
            'amount': np.random.lognormal(4, 1),
            'quantity': random.randint(1, 5),
            'payment_method': customer['preferred_payment'],
            'device_type': customer['preferred_device'],
            'discount_applied': random.choice([0, 5, 10, 15, 20]),
            'shipping_cost': random.choice([0, 50, 100, 150]),
            'city': f"City_{random.randint(1, 50)}",
            'delivery_days': random.randint(1, 7)
        }
        
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

def generate_product_data(n_products=1000):
    """
    Generate product catalog data
    """
    categories = [
        'Electronics', 'Clothing', 'Books', 'Home & Garden', 
        'Sports', 'Beauty', 'Automotive', 'Food'
    ]
    
    products = {
        'product_id': [f'PROD_{i:04d}' for i in range(1, n_products + 1)],
        'product_name': [f'Product {i}' for i in range(1, n_products + 1)],
        'category': np.random.choice(categories, n_products),
        'price': np.random.lognormal(4, 1, n_products),
        'rating': np.random.normal(4, 0.8, n_products),
        'reviews_count': np.random.poisson(50, n_products),
        'stock_quantity': np.random.poisson(100, n_products),
        'launch_date': [
            datetime.now() - timedelta(days=random.randint(30, 1095)) 
            for _ in range(n_products)
        ]
    }
    
    df = pd.DataFrame(products)
    df['rating'] = np.clip(df['rating'], 1, 5)
    df['price'] = np.clip(df['price'], 10, 10000)
    
    return df

def generate_marketing_campaign_data(n_campaigns=50):
    """
    Generate marketing campaign performance data
    """
    channels = ['Email', 'Social Media', 'Google Ads', 'Display', 'SMS', 'Direct Mail']
    campaign_types = ['Acquisition', 'Retention', 'Reactivation', 'Upsell', 'Cross-sell']
    
    campaigns = {
        'campaign_id': [f'CAMP_{i:03d}' for i in range(1, n_campaigns + 1)],
        'campaign_name': [f'Campaign {i}' for i in range(1, n_campaigns + 1)],
        'channel': np.random.choice(channels, n_campaigns),
        'campaign_type': np.random.choice(campaign_types, n_campaigns),
        'start_date': [
            datetime.now() - timedelta(days=random.randint(30, 365)) 
            for _ in range(n_campaigns)
        ],
        'budget': np.random.uniform(5000, 50000, n_campaigns),
        'impressions': np.random.poisson(100000, n_campaigns),
        'clicks': np.random.poisson(5000, n_campaigns),
        'conversions': np.random.poisson(250, n_campaigns),
        'revenue': np.random.uniform(10000, 200000, n_campaigns)
    }
    
    df = pd.DataFrame(campaigns)
    df['end_date'] = df['start_date'] + timedelta(days=30)
    df['ctr'] = (df['clicks'] / df['impressions']) * 100
    df['conversion_rate'] = (df['conversions'] / df['clicks']) * 100
    df['roas'] = df['revenue'] / df['budget']
    df['cost_per_acquisition'] = df['budget'] / df['conversions']
    
    return df

def simulate_ab_test_data(n_users=10000):
    """
    Generate A/B testing data
    """
    np.random.seed(42)
    
    # Control vs Test group
    groups = np.random.choice(['Control', 'Test'], n_users, p=[0.5, 0.5])
    
    # Different conversion rates for each group
    control_conversion_rate = 0.12
    test_conversion_rate = 0.15
    
    conversions = []
    for group in groups:
        if group == 'Control':
            conversion = np.random.random() < control_conversion_rate
        else:
            conversion = np.random.random() < test_conversion_rate
        conversions.append(int(conversion))
    
    ab_test = pd.DataFrame({
        'user_id': [f'USER_{i:05d}' for i in range(1, n_users + 1)],
        'group': groups,
        'converted': conversions,
        'session_duration': np.random.exponential(300, n_users),  # seconds
        'pages_viewed': np.random.poisson(5, n_users),
        'time_on_site': np.random.exponential(600, n_users)  # seconds
    })
    
    return ab_test

if __name__ == "__main__":
    # Generate sample datasets
    print("Generating customer data...")
    customers = generate_customer_data(5000)
    customers.to_csv('../data/customer_data.csv', index=False)
    
    print("Generating transaction data...")
    transactions = generate_transaction_data(customers, 15000)
    transactions.to_csv('../data/transaction_data.csv', index=False)
    
    print("Generating product data...")
    products = generate_product_data(1000)
    products.to_csv('../data/product_data.csv', index=False)
    
    print("Generating campaign data...")
    campaigns = generate_marketing_campaign_data(50)
    campaigns.to_csv('../data/campaign_data.csv', index=False)
    
    print("Generating A/B test data...")
    ab_test = simulate_ab_test_data(10000)
    ab_test.to_csv('../data/ab_test_data.csv', index=False)
    
    print("All datasets generated successfully!")