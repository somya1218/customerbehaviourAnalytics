import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Any

class CustomerInsightsEngine:
    """
    Advanced AI-powered customer insights generator
    """
    
    def __init__(self):
        self.insight_templates = {
            'churn_risk': [
                "🚨 {count} customers are at high risk of churning in the next 30 days",
                "⚠️ Customers who haven't purchased in {days} days show {percentage}% higher churn probability",
                "📉 {segment} segment shows concerning churn trends - immediate attention needed"
            ],
            'growth_opportunity': [
                "📈 {segment} customers have {percentage}% growth potential with targeted campaigns",
                "💰 Cross-selling opportunities could increase revenue by ${amount:,.0f}",
                "🎯 {count} customers are ready for premium tier upgrade"
            ],
            'behavioral_patterns': [
                "📱 Mobile users spend {percentage}% more than desktop users",
                "🛒 Customers purchasing {category} items show {percentage}% higher retention",
                "⏰ Peak purchasing hours are between {start_hour}-{end_hour}, optimize for better conversion"
            ],
            'segment_insights': [
                "👑 High-value customers prefer {preference} and generate {percentage}% of total revenue",
                "🆕 New customer acquisition rate has {trend} by {percentage}% this quarter",
                "🔄 Customer lifecycle average is {days} days from first to repeat purchase"
            ]
        }
    
    def generate_insights(self, data: pd.DataFrame) -> List[Dict]:
        """Generate comprehensive AI insights from customer data"""
        insights = []
        
        # Churn risk insights
        insights.extend(self._generate_churn_insights(data))
        
        # Growth opportunity insights
        insights.extend(self._generate_growth_insights(data))
        
        # Behavioral pattern insights
        insights.extend(self._generate_behavioral_insights(data))
        
        # Segment-specific insights
        insights.extend(self._generate_segment_insights(data))
        
        # Anomaly detection insights
        insights.extend(self._generate_anomaly_insights(data))
        
        return insights
    
    def _generate_churn_insights(self, data: pd.DataFrame) -> List[Dict]:
        """Generate churn-related insights"""
        insights = []
        
        if 'churn_probability' in data.columns:
            high_risk_count = len(data[data['churn_probability'] > 0.7])
            if high_risk_count > 0:
                insights.append({
                    'type': 'churn_risk',
                    'priority': 'high',
                    'title': 'High Churn Risk Alert',
                    'message': f"🚨 {high_risk_count} customers are at high risk of churning in the next 30 days",
                    'action': 'Launch retention campaign immediately',
                    'impact': f'Potential revenue loss: ${high_risk_count * data["total_spent"].mean():,.0f}',
                    'confidence': 0.85
                })
        
        # Days since last purchase insight
        if 'days_since_last_purchase' in data.columns:
            dormant_threshold = 90
            dormant_customers = data[data['days_since_last_purchase'] > dormant_threshold]
            if len(dormant_customers) > 0:
                churn_rate = dormant_customers['churn'].mean() if 'churn' in data.columns else 0.4
                insights.append({
                    'type': 'churn_risk',
                    'priority': 'medium',
                    'title': 'Dormant Customer Pattern',
                    'message': f"⚠️ Customers inactive for {dormant_threshold}+ days show {churn_rate*100:.1f}% higher churn rate",
                    'action': 'Create reactivation campaign',
                    'impact': f'Affects {len(dormant_customers)} customers',
                    'confidence': 0.78
                })
        
        return insights
    
    def _generate_growth_insights(self, data: pd.DataFrame) -> List[Dict]:
        """Generate growth opportunity insights"""
        insights = []
        
        # Cross-sell opportunities
        if 'total_orders' in data.columns and 'total_spent' in data.columns:
            low_frequency_high_value = data[
                (data['total_orders'] < data['total_orders'].median()) & 
                (data['total_spent'] > data['total_spent'].median())
            ]
            
            if len(low_frequency_high_value) > 0:
                potential_revenue = len(low_frequency_high_value) * data['avg_order_value'].mean() * 2
                insights.append({
                    'type': 'growth_opportunity',
                    'priority': 'medium',
                    'title': 'Cross-sell Opportunity',
                    'message': f"💰 {len(low_frequency_high_value)} high-value customers with low purchase frequency",
                    'action': 'Target with personalized product recommendations',
                    'impact': f'Potential additional revenue: ${potential_revenue:,.0f}',
                    'confidence': 0.72
                })
        
        # Segment upgrade opportunities
        if 'segment' in data.columns:
            upgrade_candidates = data[
                (data['segment'] == 'Medium Value') & 
                (data['total_spent'] > data[data['segment'] == 'Medium Value']['total_spent'].quantile(0.8))
            ]
            
            if len(upgrade_candidates) > 0:
                insights.append({
                    'type': 'growth_opportunity',
                    'priority': 'high',
                    'title': 'Premium Upgrade Opportunity',
                    'message': f"🎯 {len(upgrade_candidates)} customers ready for premium tier upgrade",
                    'action': 'Offer premium membership with exclusive benefits',
                    'impact': f'Potential LTV increase: 40-60%',
                    'confidence': 0.81
                })
        
        return insights
    
    def _generate_behavioral_insights(self, data: pd.DataFrame) -> List[Dict]:
        """Generate behavioral pattern insights"""
        insights = []
        
        # Device preference impact
        if 'preferred_device' in data.columns:
            device_spending = data.groupby('preferred_device')['total_spent'].mean()
            if len(device_spending) > 1:
                top_device = device_spending.idxmax()
                spending_diff = ((device_spending.max() - device_spending.min()) / device_spending.min()) * 100
                
                insights.append({
                    'type': 'behavioral_pattern',
                    'priority': 'medium',
                    'title': 'Device Usage Impact',
                    'message': f"📱 {top_device} users spend {spending_diff:.1f}% more than other device users",
                    'action': f'Optimize {top_device.lower()} experience and promote app usage',
                    'impact': 'Potential revenue increase through better UX',
                    'confidence': 0.69
                })
        
        # Category preference patterns
        if 'preferred_category' in data.columns:
            category_retention = data.groupby('preferred_category').agg({
                'churn': 'mean' if 'churn' in data.columns else lambda x: 0.2,
                'satisfaction_score': 'mean' if 'satisfaction_score' in data.columns else lambda x: 3.5
            })
            
            best_category = category_retention['satisfaction_score'].idxmax()
            retention_rate = (1 - category_retention.loc[best_category, 'churn']) * 100
            
            insights.append({
                'type': 'behavioral_pattern',
                'priority': 'low',
                'title': 'Category Performance',
                'message': f"🛒 {best_category} customers show {retention_rate:.1f}% retention rate",
                'action': f'Expand {best_category} product line and cross-promote',
                'impact': 'Improve overall customer retention',
                'confidence': 0.64
            })
        
        return insights
    
    def _generate_segment_insights(self, data: pd.DataFrame) -> List[Dict]:
        """Generate segment-specific insights"""
        insights = []
        
        if 'segment' in data.columns:
            segment_stats = data.groupby('segment').agg({
                'total_spent': ['sum', 'mean', 'count'],
                'satisfaction_score': 'mean' if 'satisfaction_score' in data.columns else lambda x: 3.5
            })
            
            # Revenue concentration
            total_revenue = data['total_spent'].sum()
            high_value_revenue = segment_stats.loc['High Value', ('total_spent', 'sum')] if 'High Value' in segment_stats.index else 0
            revenue_percentage = (high_value_revenue / total_revenue) * 100
            
            insights.append({
                'type': 'segment_insight',
                'priority': 'high',
                'title': 'Revenue Concentration',
                'message': f"👑 High-value customers generate {revenue_percentage:.1f}% of total revenue",
                'action': 'Focus retention efforts on high-value segment',
                'impact': f'Protecting ${high_value_revenue:,.0f} in revenue',
                'confidence': 0.92
            })
            
            # Segment growth trends
            if 'signup_date' in data.columns:
                recent_signups = data[data['signup_date'] > (datetime.now() - timedelta(days=90))]
                new_customer_segments = recent_signups['segment'].value_counts()
                
                if len(new_customer_segments) > 0:
                    growing_segment = new_customer_segments.idxmax()
                    growth_rate = (new_customer_segments.max() / len(recent_signups)) * 100
                    
                    insights.append({
                        'type': 'segment_insight',
                        'priority': 'medium',
                        'title': 'Segment Growth Trend',
                        'message': f"🆕 {growing_segment} segment shows strongest acquisition ({growth_rate:.1f}%)",
                        'action': f'Optimize marketing channels targeting {growing_segment} customers',
                        'impact': 'Accelerate growth in promising segments',
                        'confidence': 0.74
                    })
        
        return insights
    
    def _generate_anomaly_insights(self, data: pd.DataFrame) -> List[Dict]:
        """Generate anomaly detection insights"""
        insights = []
        
        # Unusual spending patterns
        if 'total_spent' in data.columns:
            spending_mean = data['total_spent'].mean()
            spending_std = data['total_spent'].std()
            
            # High spenders (outliers)
            high_spenders = data[data['total_spent'] > spending_mean + 2 * spending_std]
            if len(high_spenders) > 0:
                insights.append({
                    'type': 'anomaly',
                    'priority': 'medium',
                    'title': 'High-Value Outliers',
                    'message': f"💎 {len(high_spenders)} customers with exceptional spending patterns detected",
                    'action': 'Create VIP program and personalized service',
                    'impact': f'Represents ${high_spenders["total_spent"].sum():,.0f} in revenue',
                    'confidence': 0.88
                })
        
        # Satisfaction score anomalies
        if 'satisfaction_score' in data.columns and 'total_spent' in data.columns:
            # High spenders with low satisfaction
            unsatisfied_high_spenders = data[
                (data['total_spent'] > data['total_spent'].quantile(0.8)) & 
                (data['satisfaction_score'] < 3.0)
            ]
            
            if len(unsatisfied_high_spenders) > 0:
                insights.append({
                    'type': 'anomaly',
                    'priority': 'high',
                    'title': 'Satisfaction-Spending Mismatch',
                    'message': f"⚠️ {len(unsatisfied_high_spenders)} high-value customers have low satisfaction scores",
                    'action': 'Immediate customer success intervention required',
                    'impact': f'Risk of losing ${unsatisfied_high_spenders["total_spent"].sum():,.0f}',
                    'confidence': 0.91
                })
        
        return insights
    
    def get_recommendations(self, data: pd.DataFrame, segment: str = None) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if segment:
            segment_data = data[data['segment'] == segment] if 'segment' in data.columns else data
        else:
            segment_data = data
        
        # Retention recommendations
        if 'churn_probability' in segment_data.columns:
            avg_churn_prob = segment_data['churn_probability'].mean()
            if avg_churn_prob > 0.3:
                recommendations.append({
                    'category': 'Retention',
                    'priority': 'high',
                    'action': 'Launch proactive retention campaign',
                    'description': f'Target customers with churn probability > 0.5',
                    'expected_impact': 'Reduce churn by 15-25%',
                    'timeline': '2-3 weeks',
                    'resources': ['Marketing team', 'Customer success', 'Data analyst']
                })
        
        # Personalization recommendations
        if 'preferred_category' in segment_data.columns:
            recommendations.append({
                'category': 'Personalization',
                'priority': 'medium',
                'action': 'Implement category-based personalization',
                'description': 'Show relevant products based on customer preferences',
                'expected_impact': 'Increase conversion by 10-15%',
                'timeline': '4-6 weeks',
                'resources': ['Tech team', 'Product manager', 'Data scientist']
            })
        
        # Engagement recommendations
        recommendations.append({
            'category': 'Engagement',
            'priority': 'medium',
            'action': 'Create multi-channel engagement strategy',
            'description': 'Coordinate email, SMS, and push notifications',
            'expected_impact': 'Improve customer lifetime value by 20%',
            'timeline': '6-8 weeks',
            'resources': ['Marketing automation', 'Content team', 'Analytics']
        })
        
        return recommendations
    
    def generate_executive_summary(self, data: pd.DataFrame) -> Dict:
        """Generate executive summary with key metrics and insights"""
        summary = {
            'total_customers': len(data),
            'total_revenue': data['total_spent'].sum() if 'total_spent' in data.columns else 0,
            'avg_customer_value': data['total_spent'].mean() if 'total_spent' in data.columns else 0,
            'churn_rate': data['churn'].mean() * 100 if 'churn' in data.columns else 0,
            'satisfaction_score': data['satisfaction_score'].mean() if 'satisfaction_score' in data.columns else 0,
            'key_insights': [],
            'critical_actions': [],
            'growth_opportunities': []
        }
        
        # Generate insights
        all_insights = self.generate_insights(data)
        
        # Categorize insights
        for insight in all_insights:
            if insight['priority'] == 'high':
                summary['critical_actions'].append(insight['message'])
            elif insight['type'] == 'growth_opportunity':
                summary['growth_opportunities'].append(insight['message'])
            else:
                summary['key_insights'].append(insight['message'])
        
        return summary

# Convenience functions
def generate_insights(data: pd.DataFrame) -> List[Dict]:
    """Generate insights using the CustomerInsightsEngine"""
    engine = CustomerInsightsEngine()
    return engine.generate_insights(data)

def get_recommendations(data: pd.DataFrame, segment: str = None) -> List[Dict]:
    """Get recommendations using the CustomerInsightsEngine"""
    engine = CustomerInsightsEngine()
    return engine.get_recommendations(data, segment)

def generate_executive_summary(data: pd.DataFrame) -> Dict:
    """Generate executive summary using the CustomerInsightsEngine"""
    engine = CustomerInsightsEngine()
    return engine.generate_executive_summary(data)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'customer_id': [f'CUST_{i:04d}' for i in range(1, 1001)],
        'total_spent': np.random.lognormal(7, 1, 1000),
        'churn_probability': np.random.random(1000),
        'churn': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
        'segment': np.random.choice(['High Value', 'Medium Value', 'Low Value', 'At Risk'], 1000),
        'satisfaction_score': np.random.normal(3.5, 1, 1000),
        'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Books'], 1000),
        'preferred_device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 1000),
        'total_orders': np.random.poisson(5, 1000),
        'days_since_last_purchase': np.random.exponential(30, 1000)
    })
    
    engine = CustomerInsightsEngine()
    insights = engine.generate_insights(sample_data)
    recommendations = engine.get_recommendations(sample_data)
    summary = engine.generate_executive_summary(sample_data)
    
    print("Generated Insights:")
    for insight in insights[:5]:  # Show first 5 insights
        print(f"- {insight['title']}: {insight['message']}")
    
    print("\nRecommendations:")
    for rec in recommendations[:3]:  # Show first 3 recommendations
        print(f"- {rec['action']}: {rec['description']}")
    
    print(f"\nExecutive Summary:")
    print(f"Total Customers: {summary['total_customers']:,}")
    print(f"Total Revenue: ${summary['total_revenue']:,.0f}")
    print(f"Churn Rate: {summary['churn_rate']:.1f}%")