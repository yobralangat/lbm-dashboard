import os
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import warnings
from datetime import datetime
from dotenv import load_dotenv

# --- Step 1: Initial Setup (No App Initialization or Data Loading Here) ---
load_dotenv()
warnings.filterwarnings("ignore")
APP_THEME = dbc.themes.FLATLY

# --- Step 2: Define All Functions ---

def load_and_process_data():
    """Loads and processes all data from environment variable URLs."""
    print("--- RUNNING FULL DATA REFRESH PIPELINE ---")
    
    # ** THE FIX IS HERE: Get URLs from environment variables INSIDE the function **
    transactions_url = os.environ.get('TRANSACTIONS_URL')
    products_url = os.environ.get('PRODUCTS_URL')

    # Add a crucial check to see if the variables were found
    if not transactions_url or not products_url:
        print("FATAL ERROR: TRANSACTIONS_URL or PRODUCTS_URL environment variables not found.")
        # Return empty dataframes to prevent the app from crashing
        return pd.DataFrame().to_json(), pd.DataFrame().to_json(), pd.DataFrame().to_json()

    # Now, proceed with reading the data
    df_transactions = pd.read_csv(transactions_url)
    df_products = pd.read_csv(products_url)

    # (All of your data cleaning and metric calculation logic remains the same)
    df_transactions['Service Type'] = df_transactions['Service Type'].str.strip().str.lower()
    replace_map = {
        'hybrid': 'hybrid', 'hybrid  ': 'hybrid', 'classic ': 'classic', 'russian vol.': 'russian volume', 'russian volume': 'russian volume',
        'removal+hybrid': 'hybrid+removal', 'lash lift': 'lash lift', 'russ refill': 'russian refill', 'russian volume refill': 'russian volume refill',
        'hybrid +tint': 'hybrid', 'russian refill': 'russian refill', 'lash lift & tint': 'lash lift & tint', 'lash lift& tint': 'lash lift & tint',
        'classci refill': 'classic refill', 'classic +removal': 'classic+removal', 'hybrid + brow tint': 'hybrid + brow tint',
        'hybrd refill': 'hybrid refill', 'hybrd': 'hybrid', 'clasiic': 'classic', 'hybrid   ': 'hybrid', 'classic infill': 'classic refill',
        'hybrid + removal': 'hybrid+removal', 'classic + removal': 'classic+removal', 'mega refill': 'mega volume refill', 'mega + removal': 'mega volume+removal'
    }
    df_transactions['Service Type'] = df_transactions['Service Type'].replace(replace_map)
    df_transactions['Amount Paid'] = pd.to_numeric(df_transactions['Amount Paid'], errors='coerce').fillna(0)
    df_transactions['Date of Visit'] = pd.to_datetime(df_transactions['Date of Visit'], format='%d/%m/%Y', errors='coerce')
    df_transactions['Complaint'] = df_transactions['Complaint'].apply(lambda x: 1 if pd.notna(x) and str(x).strip().lower() == 'yes' else 0)
    df_transactions['Month'] = df_transactions['Date of Visit'].dt.month
    df_transactions['Year'] = df_transactions['Date of Visit'].dt.year
    df_transactions['revenue_after_vat'] = df_transactions['Amount Paid'] * (1 - 0.16)
    
    def categorize_service(st):
        ext = ['classic', 'hybrid', 'russian volume', 'refill', 'classic refill', 'hybrid refill', 'russian volume refill', 'classic+removal', 'hybrid+removal', 'removal', 'russian volume+removal', 'mega volume', 'mega volume refill', 'mega volume+removal', 'redo']
        ll = ['lash lift', 'lash lift & tint']
        if st in ext: return 'Extensions/Removals'
        if st in ll: return 'Lash Lifts'
        return None
    df_transactions['Service Category'] = df_transactions['Service Type'].apply(categorize_service)

    def calculate_commission(row):
        if row['Service Category'] == 'Extensions/Removals': return row['revenue_after_vat'] * 0.50
        if row['Service Category'] == 'Lash Lifts': return row['revenue_after_vat'] * 0.40
        return 0
    df_transactions['Commission'] = df_transactions.apply(calculate_commission, axis=1)

    df_products['DATE'] = pd.to_datetime(df_products['DATE'], format='%d/%m/%Y')
    df_products['Product Cost'] = df_products['PRICE/UNIT'] * df_products['UNITS']
    df_products['Month'] = df_products['DATE'].dt.month
    df_products['Year'] = df_products['DATE'].dt.year

    monthly_commission = df_transactions.groupby(['Artist', 'Year', 'Month'])['Commission'].sum().reset_index()
    monthly_product_cost = df_products.groupby(['ARTIST', 'Year', 'Month'])['Product Cost'].sum().reset_index()
    merged_monthly_data = pd.merge(monthly_commission, monthly_product_cost, left_on=['Artist', 'Year', 'Month'], right_on=['ARTIST', 'Year', 'Month'], how='left')
    merged_monthly_data = merged_monthly_data.drop('ARTIST', axis=1).fillna(0)
    merged_monthly_data['Net Salary'] = merged_monthly_data['Commission'] - merged_monthly_data['Product Cost']
    
    complaints = df_transactions.groupby(['Artist', 'Year', 'Month'])['Complaint'].sum().reset_index()
    redos = df_transactions[df_transactions['Service Type'] == 'redo'].groupby(['Artist', 'Year', 'Month']).size().reset_index(name='Number of Redos')
    monthly_complaints_redos = pd.merge(complaints, redos, on=['Artist', 'Year', 'Month'], how='outer').fillna(0)

    df_sorted = df_transactions.sort_values(by=['Client Name', 'Date of Visit'])
    first_visits = df_sorted.groupby('Client Name')['Date of Visit'].min().reset_index().rename(columns={'Date of Visit': 'First Visit Date'})
    df_with_first_visit = pd.merge(df_sorted, first_visits, on='Client Name', how='left')
    df_with_first_visit['First Visit Month'] = df_with_first_visit['First Visit Date'].dt.to_period('M')
    df_with_first_visit['Visit Month'] = df_with_first_visit['Date of Visit'].dt.to_period('M')
    first_visits_only = df_with_first_visit[df_with_first_visit['Visit Month'] == df_with_first_visit['First Visit Month']].drop_duplicates(subset=['Client Name', 'Artist', 'First Visit Month'])
    cohort_size = first_visits_only.groupby(['Artist', 'First Visit Month'])['Client Name'].nunique().reset_index(name='Cohort Size')
    returned_clients = df_with_first_visit[(df_with_first_visit['Visit Month'] > df_with_first_visit['First Visit Month']) & (df_with_first_visit['Visit Month'] <= (df_with_first_visit['First Visit Month'] + 3))].drop_duplicates(subset=['Client Name', 'Artist', 'First Visit Month'])
    returning_count = returned_clients.groupby(['Artist', 'First Visit Month'])['Client Name'].nunique().reset_index(name='Returning Clients')
    retention_data = pd.merge(cohort_size, returning_count, on=['Artist', 'First Visit Month'], how='left').fillna(0)
    retention_data['Retention Rate'] = (retention_data['Returning Clients'] / retention_data['Cohort Size']) * 100
    retention_data['First Visit Month'] = retention_data['First Visit Month'].astype(str)
    retention_data['MonthYear'] = pd.to_datetime(retention_data['First Visit Month'])
    
    merged_monthly_data['MonthYear'] = pd.to_datetime(merged_monthly_data[['Year', 'Month']].assign(day=1))
    monthly_complaints_redos['MonthYear'] = pd.to_datetime(monthly_complaints_redos[['Year', 'Month']].assign(day=1))

    return merged_monthly_data.to_json(date_format='iso', orient='split'), \
           monthly_complaints_redos.to_json(date_format='iso', orient='split'), \
           retention_data.to_json(date_format='iso', orient='split')

# --- Step 3: The App Factory ---
def create_dash_app():
    """Creates and configures the Dash application."""
    
    app = dash.Dash(__name__, external_stylesheets=[APP_THEME])
    
    # Load the initial data using the function
    initial_metrics_json, initial_complaints_json, initial_retention_json = load_and_process_data()

    # App Layout
    app.layout = dbc.Container(fluid=True, className="app-container", children=[
        dcc.Store(id='metrics-data-store', data=initial_metrics_json),
        dcc.Store(id='complaints-data-store', data=initial_complaints_json),
        dcc.Store(id='retention-data-store', data=initial_retention_json),
        dbc.Row(dbc.Col(html.H1("Lash Studio Performance Dashboard"), width=12, className="text-center my-4")),
        # (The rest of your layout is the same)
        dbc.Row([
            dbc.Col(html.H5(id='live-clock', className="text-start"), width=6),
            dbc.Col(dbc.Button("Refresh Data", id="refresh-button", n_clicks=0, color="primary", className="float-end"), width=6)
        ], className="mb-4"),
        dcc.Interval(id='interval-clock', interval=1000),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H5("Select Artist", className="card-title"),
                dcc.Dropdown(id='artist-dropdown', value='All')
            ])]), md=6, className="mb-4"),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H5("Select Date Range", className="card-title"),
                dcc.DatePickerRange(id='date-range-picker', display_format='MMM YYYY', className="w-100")
            ])]), md=6, className="mb-4"),
        ]),
        html.Div(id='dashboard-content')
    ])

    # Register Callbacks
    register_callbacks(app)
    
    return app

def register_callbacks(app):
    """Registers all the app's callbacks."""
    # (All your callbacks remain exactly the same as the last working version)
    # They will be attached to the 'app' instance here.
    @app.callback(...)
    def refresh_data_and_store(n_clicks): ...

    @app.callback(...)
    def update_controls(metrics_json): ...

    @app.callback(...)
    def update_main_dashboard(...): ...

    @app.callback(...)
    def update_clock(n): ...

# --- Step 4: Create and Run the App ---
app = create_dash_app()
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)