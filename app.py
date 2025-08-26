import os
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import warnings
from datetime import datetime
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# --- Step 1: Initial Setup ---
load_dotenv()
warnings.filterwarnings("ignore")
APP_THEME = dbc.themes.QUARTZ

# --- Step 2: Define Data Processing Logic ---
def load_and_process_data():
    """
    Loads and processes all data from Google Sheets using the official API.
    This is the most robust way to fetch the data.
    """
    print("--- RUNNING FULL DATA REFRESH PIPELINE VIA API ---")
    
    # Get credentials and URLs from environment variables INSIDE the function
    # This is crucial for deployment stability.
    google_creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    transactions_url = os.environ.get('TRANSACTIONS_URL')
    products_url = os.environ.get('PRODUCTS_URL')

    if not all([google_creds_json_str, transactions_url, products_url]):
        print("FATAL ERROR: Missing one or more required environment variables.")
        # Return empty dataframes as JSON to prevent the app from crashing
        return pd.DataFrame().to_json(), pd.DataFrame().to_json(), pd.DataFrame().to_json()

    # Authenticate with Google Sheets API
    creds_dict = json.loads(google_creds_json_str)
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    print("Opening sheets via API...")
    transactions_sheet = client.open_by_url(transactions_url).sheet1
    products_sheet = client.open_by_url(products_url).sheet1
    
    print("Fetching all records...")
    df_transactions = pd.DataFrame(transactions_sheet.get_all_records())
    df_products = pd.DataFrame(products_sheet.get_all_records())
    
    print("Data fetched. Starting cleaning and processing...")
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
    df_products['DATE'] = pd.to_datetime(df_products['DATE'], format='%d/%m/%Y', errors='coerce')
    df_products['PRICE/UNIT'] = pd.to_numeric(df_products['PRICE/UNIT'], errors='coerce').fillna(0)
    df_products['UNITS'] = pd.to_numeric(df_products['UNITS'], errors='coerce').fillna(0)
    df_products['Product Cost'] = df_products['PRICE/UNIT'] * df_products['UNITS']
    df_products['Month'] = df_products['DATE'].dt.month
    df_products['Year'] = df_products['DATE'].dt.year

    # --- Calculate Final Metrics ---
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
    
    initial_metrics_json, initial_complaints_json, initial_retention_json = load_and_process_data()

    app.layout = dbc.Container(fluid=True, className="app-container", children=[
        dcc.Store(id='metrics-data-store', data=initial_metrics_json),
        dcc.Store(id='complaints-data-store', data=initial_complaints_json),
        dcc.Store(id='retention-data-store', data=initial_retention_json),
        dbc.Row(dbc.Col(html.H1("Lash Studio Performance Dashboard"), width=12, className="text-center my-4")),
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

    register_callbacks(app)
    return app

def register_callbacks(app):
    """Registers all the app's callbacks."""
    @app.callback(
        Output('metrics-data-store', 'data', allow_duplicate=True),
        Output('complaints-data-store', 'data', allow_duplicate=True),
        Output('retention-data-store', 'data', allow_duplicate=True),
        Input('refresh-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def refresh_data_and_store(n_clicks):
        if n_clicks > 0:
            return load_and_process_data()
        return dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output('artist-dropdown', 'options'),
        Output('date-range-picker', 'min_date_allowed'),
        Output('date-range-picker', 'max_date_allowed'),
        Output('date-range-picker', 'start_date'),
        Output('date-range-picker', 'end_date'),
        Input('metrics-data-store', 'data')
    )
    def update_controls(metrics_json):
        if not metrics_json or pd.read_json(metrics_json, orient='split').empty:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        df = pd.read_json(metrics_json, orient='split')
        df['MonthYear'] = pd.to_datetime(df['MonthYear'])
        unique_artists = sorted(df['Artist'].unique())
        artist_options = [{'label': 'All Artists', 'value': 'All'}] + [{'label': artist, 'value': artist} for artist in unique_artists]
        min_date, max_date = df['MonthYear'].min().date(), df['MonthYear'].max().date()
        return artist_options, min_date, max_date, min_date, max_date

    @app.callback(
        Output('dashboard-content', 'children'),
        Input('artist-dropdown', 'value'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date'),
        State('metrics-data-store', 'data'),
        State('complaints-data-store', 'data'),
        State('retention-data-store', 'data')
    )
    def update_main_dashboard(selected_artist, start_date_str, end_date_str, metrics_json, complaints_json, retention_json):
        if not all([selected_artist, start_date_str, end_date_str, metrics_json, complaints_json, retention_json]):
            return "" 
        
        merged_monthly_data = pd.read_json(metrics_json, orient='split')
        monthly_complaints_redos = pd.read_json(complaints_json, orient='split')
        retention_data = pd.read_json(retention_json, orient='split')
        if merged_monthly_data.empty:
             return dbc.Alert("No data available to display.", color="warning")

        merged_monthly_data['MonthYear'] = pd.to_datetime(merged_monthly_data['MonthYear'])
        monthly_complaints_redos['MonthYear'] = pd.to_datetime(monthly_complaints_redos['MonthYear'])
        retention_data['MonthYear'] = pd.to_datetime(retention_data['MonthYear'])
        start_date, end_date = pd.to_datetime(start_date_str), pd.to_datetime(end_date_str)

        metrics_by_date = merged_monthly_data[(merged_monthly_data['MonthYear'] >= start_date) & (merged_monthly_data['MonthYear'] <= end_date)]
        complaints_by_date = monthly_complaints_redos[(monthly_complaints_redos['MonthYear'] >= start_date) & (monthly_complaints_redos['MonthYear'] <= end_date)]
        retention_by_date = retention_data[(retention_data['MonthYear'] >= start_date) & (retention_data['MonthYear'] <= end_date)]
        
        if selected_artist == 'All':
            title_name = "All Artists (Studio Total)"
            metrics_display_df = metrics_by_date.groupby('MonthYear').agg(Commission=('Commission', 'sum'), **{'Net Salary': ('Net Salary', 'sum')}).reset_index()
            complaints_display_df = complaints_by_date.groupby('MonthYear').agg(Complaint=('Complaint', 'sum'), **{'Number of Redos': ('Number of Redos', 'sum')}).reset_index()
            retention_display_df = retention_by_date.groupby('MonthYear').agg(**{'Retention Rate': ('Retention Rate', 'mean')}).reset_index()
        else:
            title_name = selected_artist
            metrics_display_df = metrics_by_date[metrics_by_date['Artist'] == selected_artist]
            complaints_display_df = complaints_by_date[complaints_by_date['Artist'] == selected_artist]
            retention_display_df = retention_by_date[retention_by_date['Artist'] == selected_artist]

        if metrics_display_df.empty:
            return dbc.Alert(f"No data available for {title_name} in the selected date range.", color="info", className="m-4")

        total_commission = int(metrics_display_df['Commission'].sum())
        total_net_salary = int(metrics_display_df['Net Salary'].sum())
        total_complaints = int(complaints_display_df['Complaint'].sum())
        total_redos = int(complaints_display_df['Number of Redos'].sum())
        avg_retention = float(retention_display_df['Retention Rate'].mean()) if not retention_display_df.empty else 0.0

        color_arg = {'color': 'Artist'} if 'Artist' in metrics_display_df.columns else {}
        fig_commission = px.line(metrics_display_df, x='MonthYear', y='Commission', title=f'Commission Trend for {title_name}', markers=True, **color_arg)
        fig_net_salary = px.line(metrics_display_df, x='MonthYear', y='Net Salary', title=f'Net Salary Trend for {title_name}', markers=True, **color_arg)
        fig_complaints = px.bar(complaints_display_df, x='MonthYear', y=['Complaint', 'Number of Redos'], title=f'Complaints & Redos for {title_name}', barmode='group')
        fig_retention = px.line(retention_display_df, x='MonthYear', y='Retention Rate', title=f'Client Retention Rate for {title_name}', markers=True, **color_arg)
        
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"Ksh {total_commission:,.0f}"), html.P("Total Commission")])]), md=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"Ksh {total_net_salary:,.0f}"), html.P("Total Net Salary")])]), md=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{total_complaints}"), html.P("Total Complaints")])]), md=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{total_redos}"), html.P("Total Redos")])]), md=3),
            ], className="text-center mb-4"),
            dbc.Row([
                 dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{avg_retention:.1f}%"), html.P("Avg. Retention Rate in Period")])]), md=3, className="mx-auto")
            ], className="text-center mb-4"),
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(figure=fig_commission)), md=6, className="mb-4"),
                dbc.Col(dbc.Card(dcc.Graph(figure=fig_net_salary)), md=6, className="mb-4"),
                dbc.Col(dbc.Card(dcc.Graph(figure=fig_complaints)), md=6, className="mb-4"),
                dbc.Col(dbc.Card(dcc.Graph(figure=fig_retention)), md=6, className="mb-4"),
            ]),
            dbc.Accordion([
                dbc.AccordionItem(dbc.Table.from_dataframe(metrics_display_df.round(2), striped=True, bordered=True, hover=True), title="Monthly Salary Data"),
                dbc.AccordionItem(dbc.Table.from_dataframe(retention_display_df.round(2), striped=True, bordered=True, hover=True), title="Client Retention Data"),
                dbc.AccordionItem(dbc.Table.from_dataframe(complaints_display_df.round(2), striped=True, bordered=True, hover=True), title="Complaints & Redos Data"),
            ], start_collapsed=True)
        ])

    @app.callback(
        Output('live-clock', 'children'),
        Input('interval-clock', 'n_intervals')
    )
    def update_clock(n):
        return f"Live Report as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# --- Step 4: Create and Run the App ---
app = create_dash_app()
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)