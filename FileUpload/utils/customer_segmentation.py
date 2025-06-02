import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import re
import os
warnings.filterwarnings('ignore')

np.random.seed(42)

def find_column(df, possible_names, case_sensitive=False):
    for col in df.columns:
        col_clean = col if case_sensitive else col.lower()
        for name in possible_names:
            name_clean = name if case_sensitive else name.lower()
            if col_clean == name_clean or re.sub(r'[\W_]', '', col_clean) == re.sub(r'[\W_]', '', name_clean):
                return col
    return None

def load_and_preprocess_data(file_path, column_mapping=None):
    try:
        default_mapping = {
            'InvoiceNo': ['invoiceno', 'invoice_no', 'invoice', 'order_id', 'order_no'],
            'StockCode': ['stockcode', 'stock_code', 'product_id', 'item_code'],
            'Description': ['description', 'product_description', 'item_name'],
            'Quantity': ['quantity', 'qty', 'amount'],
            'InvoiceDate': ['invoicedate', 'invoice_date', 'date', 'order_date', 'transaction_date'],
            'UnitPrice': ['unitprice', 'unit_price', 'price', 'cost'],
            'CustomerID': ['customerid', 'customer_id', 'client_id', 'user_id'],
            'Country': ['country', 'location', 'region']
        }
        df = pd.read_csv(file_path)
        if column_mapping is None:
            column_mapping = {}
            for key, possibles in default_mapping.items():
                found_col = find_column(df, possibles)
                if found_col:
                    column_mapping[key] = found_col
                else:
                    print(f"Warning: No matching column found for {key}")
                    return None
        df = df.rename(columns={v: k for k, v in column_mapping.items() if v in df.columns})
        required_cols = ['InvoiceNo', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return None
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df = df.dropna(subset=['InvoiceDate', 'CustomerID'])
        df['CustomerID'] = df['CustomerID'].astype(int)
        df = df[df['Quantity'] > 0]
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        df = df[df['TotalPrice'] > 0]
        if 'Description' in df.columns:
            df = df.dropna(subset=['Description'])
            df['Description'] = df['Description'].str.strip().str.upper()
            df = df[df['Description'] != '']
            print(f"Cleaned Description column: {len(df)} rows remaining")
        return df
    except Exception as e:
        print(f"Error in data loading/preprocessing: {e}")
        return None

def analyze_top_products(df, top_n=10, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    product_cols = ['StockCode', 'Description'] if 'Description' in df.columns else ['StockCode']
    product_summary = df.groupby(product_cols).agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).sort_values(by='Quantity', ascending=False).head(top_n)
    product_summary.to_csv(os.path.join(output_dir, 'top_products.csv'))

    plt.figure(figsize=(10, 6))
    plot = sns.barplot(x='Quantity', y=product_summary.index.get_level_values('Description'),
                       data=product_summary.reset_index())
    plt.title('Top Products by Quantity Sold')
    plt.xlabel('Quantity Sold')
    plt.ylabel('Product')
    for i, v in enumerate(product_summary['Quantity']):
        plot.text(v * 1.02, i, f'{int(v)}', va='center', ha='left')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'top_products_quantity.png')
    plt.savefig(plot_path)
    plt.close()

    return product_summary, plot_path

def calculate_rfm(df, reference_date):
    reference_date = pd.to_datetime(reference_date)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]
    return rfm

def segment_customers(rfm):
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
    rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
    rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    rfm['Segment'] = 'Low Value'
    rfm.loc[rfm['Cluster'] == rfm[rfm['Monetary'] > rfm['Monetary'].median()]['Cluster'].iloc[0], 'Segment'] = 'High Value'
    rfm.loc[rfm['Cluster'] == rfm[rfm['Frequency'] > rfm['Frequency'].median()]['Cluster'].iloc[0], 'Segment'] = 'Loyal'
    rfm.loc[rfm['Cluster'] == rfm[rfm['Recency'] > rfm['Recency'].median()]['Cluster'].iloc[0], 'Segment'] = 'Inactive'
    return rfm

def visualize_segments(rfm, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', size='Frequency',
                    sizes=(50, 500), alpha=0.6)
    plt.title('Customer Segments (RFM Analysis)')
    plt.xlabel('Recency (days since last purchase)')
    plt.ylabel('Monetary (total spending)')
    plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'customer_segments.png')
    plt.savefig(plot_path)
    plt.close()

    plt.figure(figsize=(8, 6))
    plot = sns.countplot(data=rfm, x='Segment')
    plt.title('Customer Segment Distribution')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    for p in plot.patches:
        plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom')
    plt.tight_layout()
    segment_dist_path = os.path.join(output_dir, 'segment_distribution.png')
    plt.savefig(segment_dist_path)
    plt.close()

    return [plot_path, segment_dist_path]

def analyze_loyalty(df, rfm, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    loyalty_analysis = df.merge(rfm[['CustomerID', 'Segment']], on='CustomerID')
    loyalty_summary = loyalty_analysis.groupby('Segment').agg({
        'TotalPrice': 'mean',
        'Quantity': 'mean',
        'InvoiceNo': 'nunique'
    }).reset_index()
    loyalty_summary.columns = ['Segment', 'AvgPurchaseValue', 'AvgQuantity', 'UniqueInvoices']
    loyalty_summary.to_csv(os.path.join(output_dir, 'loyalty_summary.csv'))
    return loyalty_summary

def market_basket_analysis(df, min_support=0.005, min_confidence=0.1, output_dir='.', sample_size=10000):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting market basket analysis with {len(df)} rows")
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows for market basket analysis")
    try:
        basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('InvoiceNo'))
        basket = basket.apply(lambda x: (x > 0).astype(int))
        print(f"Basket matrix created with shape: {basket.shape}")
    except Exception as e:
        print(f"Error creating basket matrix: {e}")
        return pd.DataFrame(), None
    try:
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True, low_memory=True)
        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        if frequent_itemsets.empty:
            print("No frequent itemsets found. Try lowering min_support.")
            return pd.DataFrame(), None
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
        print(f"Generated {len(rules)} association rules")
    except Exception as e:
        print(f"Error in Apriori algorithm: {e}")
        return pd.DataFrame(), None
    if rules.empty:
        print("No association rules generated.")
        return pd.DataFrame(), None
    rules.to_csv(os.path.join(output_dir, 'association_rules.csv'))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rules, x='support', y='confidence', size='lift', sizes=(50, 500), alpha=0.6)
    plt.title('Association Rules: Support vs Confidence')
    plt.xlabel('Support (frequency of pair)')
    plt.ylabel('Confidence (likelihood of pair)')
    plt.legend(title='Lift', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'association_rules.png')
    plt.savefig(plot_path)
    plt.close()

    return rules, plot_path

def calculate_clv(df, rfm, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    clv = rfm[['CustomerID', 'Monetary', 'Frequency']].copy()
    clv['CLV'] = clv['Monetary'] * clv['Frequency']
    clv = clv.merge(rfm[['CustomerID', 'Segment']], on='CustomerID')
    clv.to_csv(os.path.join(output_dir, 'customer_clv.csv'))

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=clv, x='Segment', y='CLV')
    plt.title('Customer Lifetime Value by Segment')
    plt.xlabel('Segment')
    plt.ylabel('CLV (estimated future revenue)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'clv_by_segment.png')
    plt.savefig(plot_path)
    plt.close()

    return clv, plot_path

def geographical_analysis(df, top_n=10, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    if 'Country' not in df.columns:
        print("No Country column found.")
        return pd.DataFrame(), None
    geo_summary = df.groupby('Country').agg({
        'TotalPrice': 'sum',
        'Quantity': 'sum',
        'CustomerID': 'nunique'
    }).reset_index().sort_values(by='TotalPrice', ascending=False).head(top_n)
    geo_summary.columns = ['Country', 'TotalRevenue', 'TotalQuantity', 'UniqueCustomers']
    geo_summary.to_csv(os.path.join(output_dir, 'geo_summary.csv'))

    plt.figure(figsize=(10, 6))
    plot = sns.barplot(data=geo_summary, x='TotalRevenue', y='Country')
    plt.title('Top Countries by Revenue')
    plt.xlabel('Total Revenue')
    plt.ylabel('Country')
    for i, v in enumerate(geo_summary['TotalRevenue']):
        plot.text(v * 1.02, i, f'${v:,.0f}', va='center', ha='left')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'geo_sales.png')
    plt.savefig(plot_path)
    plt.close()

    return geo_summary, plot_path

def purchase_timing_analysis(df, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    try:
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        df['Hour'] = df['InvoiceDate'].dt.hour
    except Exception as e:
        print(f"Error processing dates: {e}")
        return pd.DataFrame(), pd.DataFrame(), []
    day_summary = df.groupby('DayOfWeek').agg({
        'TotalPrice': 'sum'
    }).reset_index().sort_values(by='TotalPrice', ascending=False)
    day_summary.to_csv(os.path.join(output_dir, 'sales_by_day.csv'))

    plt.figure(figsize=(8, 6))
    plot = sns.barplot(data=day_summary, x='DayOfWeek', y='TotalPrice')
    plt.title('Sales by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Total Revenue')
    for p in plot.patches:
        plot.annotate(f'${p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.tight_layout()
    day_plot_path = os.path.join(output_dir, 'sales_by_day.png')
    plt.savefig(day_plot_path)
    plt.close()

    hour_summary = df.groupby('Hour').agg({
        'TotalPrice': 'sum'
    }).reset_index()
    hour_summary.to_csv(os.path.join(output_dir, 'sales_by_hour.csv'))

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=hour_summary, x='Hour', y='TotalPrice', marker='o')
    plt.title('Sales by Hour of Day')
    plt.xlabel('Hour (24-hour format)')
    plt.ylabel('Total Revenue')
    plt.grid(True)
    plt.tight_layout()
    hour_plot_path = os.path.join(output_dir, 'sales_by_hour.png')
    plt.savefig(hour_plot_path)
    plt.close()

    return day_summary, hour_summary, [day_plot_path, hour_plot_path]

def churn_analysis(rfm, recency_threshold=180, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    churn = rfm[['CustomerID', 'Recency', 'Segment']].copy()
    churn['ChurnRisk'] = churn['Recency'] > recency_threshold
    churn.to_csv(os.path.join(output_dir, 'churn_analysis.csv'))

    plt.figure(figsize=(8, 6))
    plot = sns.countplot(data=churn, x='Segment', hue='ChurnRisk')
    plt.title('Churn Risk by Customer Segment')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    for p in plot.patches:
        plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom')
    plt.legend(title='Churn Risk')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'churn_risk.png')
    plt.savefig(plot_path)
    plt.close()

    return churn, plot_path

def run_analysis(file_path, output_dir='plots', reference_date='2025-06-02', column_mapping=None,
                 min_support=0.005, min_confidence=0.1, sample_size=10000):
    plot_dir = os.path.join('C:/CustomerSegmentation', 'media', output_dir)
    csv_dir = os.path.join('C:/CustomerSegmentation', 'media', 'outputs')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    df = load_and_preprocess_data(file_path, column_mapping)
    if df is None:
        print("Data loading failed: No valid data returned.")
        return None
    top_products, top_products_plot = analyze_top_products(df, output_dir=plot_dir)
    rfm = calculate_rfm(df, reference_date)
    rfm_segmented = segment_customers(rfm)
    rfm_segmented.to_csv(os.path.join(plot_dir, 'rfm_segments.csv'))
    segment_plots = visualize_segments(rfm_segmented, output_dir=plot_dir)
    loyalty_summary = analyze_loyalty(df, rfm_segmented, output_dir=plot_dir)
    rules, rules_plot = (market_basket_analysis(df, min_support=min_support, min_confidence=min_confidence,
                                                output_dir=plot_dir, sample_size=sample_size)
                         if 'Description' in df.columns else (pd.DataFrame(), None))
    clv, clv_plot = calculate_clv(df, rfm_segmented, output_dir=plot_dir)
    geo_summary, geo_plot = (geographical_analysis(df, output_dir=plot_dir)
                             if 'Country' in df.columns else (pd.DataFrame(), None))
    day_summary, hour_summary, timing_plots = purchase_timing_analysis(df, output_dir=plot_dir)
    churn, churn_plot = churn_analysis(rfm_segmented, output_dir=plot_dir)

    plots = [top_products_plot, rules_plot, clv_plot, geo_plot, churn_plot] + segment_plots + timing_plots
    plots = ['/media/plots/' + os.path.basename(p) for p in plots if p]

    return {
        'top_products': top_products.to_dict('records'),
        'rfm_segments': rfm_segmented.to_dict('records'),
        'loyalty_summary': loyalty_summary.to_dict('records'),
        'rules': rules.to_dict('records') if not rules.empty else [],
        'clv': clv.to_dict('records'),
        'geo_summary': geo_summary.to_dict('records') if not geo_summary.empty else [],
        'day_summary': day_summary.to_dict('records'),
        'hour_summary': hour_summary.to_dict('records'),
        'churn': churn.to_dict('records'),
        'plots': plots
    }