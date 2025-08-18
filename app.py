import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from datetime import datetime
from dotenv import load_dotenv

# --- Configuration & Setup ---
load_dotenv()
warnings.filterwarnings("ignore")
APP_THEME = dbc.themes.FLATLY

# Load environment variables
transactions_url = os.environ['TRANSACTIONS_URL']
products_url = os.environ['PRODUCTS_URL']

# --- Data Processing Functions (Refactored to be self-contained) ---

def load_and_process_data():
    """
    This is the main data pipeline function. It loads all data, cleans it,
    and returns the final dataframes needed for the dashboard.
    """
    print("--- RUNNING FULL DATA REFRESH PIPELINE ---")
    
    # Load raw data
    df_transactions = pd.read_csv(transactions_url)
    df_products = pd.read_csv(products_url)

    # --- Clean Transactions ---
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

    # Convert 'Complaint' column from text ('Yes') to a numeric flag (1 or 0)
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

    # --- Clean Products ---
    df_products['DATE'] = pd.to_datetime(df_products['DATE'], format='%d/%m/%Y')
    df_products['Product Cost'] = df_products['PRICE/UNIT'] * df_products['UNITS']
    df_products['Month'] = df_products['DATE'].dt.month
    df_products['Year'] = df_products['DATE'].dt.year

    # --- Calculate Final Metrics ---
    monthly_commission = df_transactions.groupby(['Artist', 'Year', 'Month'])['Commission'].sum().reset_index()
    monthly_product_cost = df_products.groupby(['ARTIST', 'Year', 'Month'])['Product Cost'].sum().reset_index()
    merged_monthly_data = pd.merge(monthly_commission, monthly_product_cost, left_on=['Artist', 'Year', 'Month'], right_on=['ARTIST', 'Year', 'Month'], how='left')
    merged_monthly_data = merged_monthly_data.drop('ARTIST', axis=1).fillna(0)
    merged_monthly_data['Net Salary'] = merged_monthly_data['Commission'] - merged_monthly_data['Product Cost']
    
    # Calculate complaints and redos from the cleaned transactions data
    complaints = df_transactions.groupby(['Artist', 'Year', 'Month'])['Complaint'].sum().reset_index()
    redos = df_transactions[df_transactions['Service Type'] == 'redo'].groupby(['Artist', 'Year', 'Month']).size().reset_index(name='Number of Redos')
    monthly_complaints_redos = pd.merge(complaints, redos, on=['Artist', 'Year', 'Month'], how='outer').fillna(0)

    # Create MonthYear for joining
    merged_monthly_data['MonthYear'] = pd.to_datetime(merged_monthly_data[['Year', 'Month']].assign(day=1))
    monthly_complaints_redos['MonthYear'] = pd.to_datetime(monthly_complaints_redos[['Year', 'Month']].assign(day=1))

    # Convert to JSON for storage
    return merged_monthly_data.to_json(date_format='iso', orient='split'), \
           monthly_complaints_redos.to_json(date_format='iso', orient='split')

# --- Initialize the Dash App ---
app = dash.Dash(__name__, external_stylesheets=[APP_THEME])
server = app.server

# --- App Layout ---
app.layout = dbc.Container(fluid=True, className="app-container", children=[
    # Hidden stores for data
    dcc.Store(id='metrics-data-store'),
    dcc.Store(id='complaints-data-store'),
    
    dbc.Row(dbc.Col(html.H1("Artists' Performance Dashboard"), width=12, className="text-center my-4")),
    dbc.Row([
        dbc.Col(html.H5(id='live-clock', className="text-start"), width=6),
        dbc.Col(dbc.Button("Refresh Data", id="refresh-button", n_clicks=0, color="primary", className="float-end"), width=6)
    ], className="mb-4"),

    dcc.Interval(id='interval-clock', interval=1*1000, n_intervals=0),
    
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H5("Select Artist", className="card-title"),
            dcc.Dropdown(id='artist-dropdown', value='All') # Options will be set by a callback
        ])]), md=6, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H5("Select Date Range", className="card-title"),
            dcc.DatePickerRange(id='date-range-picker', display_format='MMM YYYY', className="w-100")
        ])]), md=6, className="mb-4"),
    ]),
    
    html.Div(id='dashboard-content') 
])

# --- CALLBACK 1: Refresh data and store it ---
@app.callback(
    Output('metrics-data-store', 'data'),
    Output('complaints-data-store', 'data'),
    Input('refresh-button', 'n_clicks')
)
def refresh_data_and_store(n_clicks):
    metrics_json, complaints_json = load_and_process_data()
    return metrics_json, complaints_json

# --- CALLBACK 2: Update controls based on stored data ---
# --- CALLBACK 2: Update controls based on stored data ---
@app.callback(
    Output('artist-dropdown', 'options'),
    Output('date-range-picker', 'min_date_allowed'),
    Output('date-range-picker', 'max_date_allowed'),
    Output('date-range-picker', 'start_date'),
    Output('date-range-picker', 'end_date'),
    Input('metrics-data-store', 'data')
)
def update_controls(metrics_json):
    if not metrics_json:
        # This callback has 5 Outputs, so it must return 5 values.
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    df = pd.read_json(metrics_json, orient='split')
    unique_artists = sorted(df['Artist'].unique())
    artist_options = [{'label': 'All Artists', 'value': 'All'}] + [{'label': artist, 'value': artist} for artist in unique_artists]
    
    min_date = df['MonthYear'].min().date()
    max_date = df['MonthYear'].max().date()
    
    return artist_options, min_date, max_date, min_date, max_date

# --- CALLBACK 3: Update main dashboard based on controls AND stored data ---
@app.callback(
    Output('dashboard-content', 'children'),
    Input('artist-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    State('metrics-data-store', 'data'),
    State('complaints-data-store', 'data')
)
def update_main_dashboard(selected_artist, start_date_str, end_date_str, metrics_json, complaints_json):
    if not all([selected_artist, start_date_str, end_date_str, metrics_json, complaints_json]):
        return "" 

    # Read data from the store
    merged_monthly_data = pd.read_json(metrics_json, orient='split')
    monthly_complaints_redos = pd.read_json(complaints_json, orient='split')
    
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Filter by date
    metrics_by_date = merged_monthly_data[(merged_monthly_data['MonthYear'] >= start_date) & (merged_monthly_data['MonthYear'] <= end_date)]
    complaints_by_date = monthly_complaints_redos[(monthly_complaints_redos['MonthYear'] >= start_date) & (monthly_complaints_redos['MonthYear'] <= end_date)]
    
    # Filter by artist
    if selected_artist == 'All':
        title_name = "All Artists (Studio Total)"
        metrics_display_df = metrics_by_date.groupby('MonthYear').agg(Commission=('Commission', 'sum'), **{'Net Salary': ('Net Salary', 'sum')}).reset_index()
        complaints_display_df = complaints_by_date.groupby('MonthYear').agg(Complaint=('Complaint', 'sum'), **{'Number of Redos': ('Number of Redos', 'sum')}).reset_index()
    else:
        title_name = selected_artist
        metrics_display_df = metrics_by_date[metrics_by_date['Artist'] == selected_artist]
        complaints_display_df = complaints_by_date[complaints_by_date['Artist'] == selected_artist]

    if metrics_display_df.empty:
        return dbc.Alert(f"No data available for {title_name} in the selected date range.", color="info", className="m-4")

    # KPIs
    total_commission = int(metrics_display_df['Commission'].sum())
    total_net_salary = int(metrics_display_df['Net Salary'].sum())
    total_complaints = int(complaints_display_df['Complaint'].sum())
    total_redos = int(complaints_display_df['Number of Redos'].sum())

    # Figures
    color_arg = {'color': 'Artist'} if 'Artist' in metrics_display_df.columns else {}
    fig_commission = px.line(metrics_display_df, x='MonthYear', y='Commission', title=f'Commission Trend for {title_name}', markers=True, **color_arg)
    fig_net_salary = px.line(metrics_display_df, x='MonthYear', y='Net Salary', title=f'Net Salary Trend for {title_name}', markers=True, **color_arg)
    fig_complaints = px.bar(complaints_display_df, x='MonthYear', y=['Complaint', 'Number of Redos'], title=f'Complaints & Redos for {title_name}', barmode='group')

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"Ksh {total_commission:,.0f}"), html.P("Total Commission")])]), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"Ksh {total_net_salary:,.0f}"), html.P("Total Net Salary")])]), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{total_complaints}"), html.P("Total Complaints")])]), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{total_redos}"), html.P("Total Redos")])]), md=3),
        ], className="text-center mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_commission)), md=6, className="mb-4"),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_net_salary)), md=6, className="mb-4"),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_complaints)), md=6, className="mb-4"),
        ])
    ])

# --- Clock Callback (no change) ---
@app.callback(
    Output('live-clock', 'children'),
    Input('interval-clock', 'n_intervals')
)
def update_clock(n):
    return f"Live Report as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


if __name__ == '__main__':
    app.run_server(debug=True)```

### **Key Changes in This Version**

1.  **`load_and_process_data` Restored:** This function now correctly calculates and returns both the `merged_monthly_data` and the `monthly_complaints_redos` dataframes.
2.  **Complaint Calculation Re-instated:** It correctly reads the 'Complaint' column from your `Transactions` sheet and aggregates the totals.
3.  **Two `dcc.Store` Components:** The layout now has `metrics-data-store` and `complaints-data-store` to hold the two separate streams of data.
4.  **Callbacks Updated:**
    *   `refresh_data_and_store` now populates both stores.
    *   `update_main_dashboard` now takes both stores as `State` and uses the data to build the full dashboard, including the restored complaint/redo KPIs and chart.

This version accurately reflects your request: it uses the stable `dcc.Store` architecture while keeping all the complaint-related analysis that is available from your primary transaction sheet.