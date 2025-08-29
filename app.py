import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table
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
APP_THEME = dbc.themes.LUX

# --- Step 2: Define Data Processing Logic ---
def load_and_process_data():
    """Loads and processes all data from Google Sheets using the official API."""
    print("--- RUNNING FULL DATA REFRESH PIPELINE VIA API ---")
    
    google_creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    transactions_url = os.environ.get('TRANSACTIONS_URL')
    products_url = os.environ.get('PRODUCTS_URL')

    if not all([google_creds_json_str, transactions_url, products_url]):
        print("FATAL ERROR: Missing one or more required environment variables.")
        empty_json = pd.DataFrame().to_json(date_format='iso', orient='split')
        return empty_json, empty_json, empty_json, empty_json, empty_json

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

    # --- ** NEW, ROBUST STANDARDIZATION FUNCTION ** ---
    def standardize_service_name(service_name):
        if not isinstance(service_name, str):
            return "Unknown"
        
        name = service_name.lower().strip().replace(" ", "")
        
        if "+removal" in name:
            if "classic" in name: return "Classic + Removal"
            if "hybrid" in name: return "Hybrid + Removal"
            if "russian" in name or "volume" in name: return "Volume + Removal"
            return "Removal"
        if "refill" in name:
            if "classic" in name: return "Classic Refill"
            if "hybrid" in name: return "Hybrid Refill"
            if "russian" in name or "volume" in name: return "Volume Refill"
            return "Refill"
        if "classic" in name: return "Classic Full Set"
        if "hybrid" in name: return "Hybrid Full Set"
        if "russian" in name or "vol." in name or "volume" in name: return "Volume Full Set"
        if "lashlift" in name or "lift" in name: return "Lash Lift"
        if "redo" in name: return "Redo"
        return service_name.title()

    df_transactions['Service Type'] = df_transactions['Service Type'].apply(standardize_service_name)
    
    df_transactions['Amount Paid'] = pd.to_numeric(df_transactions['Amount Paid'], errors='coerce').fillna(0)
    df_transactions['Date of Visit'] = pd.to_datetime(df_transactions['Date of Visit'], format='%d/%m/%Y', errors='coerce')
    df_transactions['Complaint'] = df_transactions['Complaint'].apply(lambda x: 1 if pd.notna(x) and str(x).strip().lower() == 'yes' else 0)
    df_transactions['Month'] = df_transactions['Date of Visit'].dt.month
    df_transactions['Year'] = df_transactions['Date of Visit'].dt.year
    df_transactions['revenue_after_vat'] = df_transactions['Amount Paid'] * (1 - 0.16)
    
    def categorize_service(st):
        if not isinstance(st, str): return None
        st_lower = st.lower()
        if "lift" in st_lower: return 'Lash Lifts'
        if any(keyword in st_lower for keyword in ['classic', 'hybrid', 'volume', 'refill', 'removal', 'redo']):
            return 'Extensions/Removals'
        return None
    df_transactions['Service Category'] = df_transactions['Service Type'].apply(categorize_service)

    def calculate_commission(row):
        if row['Service Category'] == 'Extensions/Removals': return row['revenue_after_vat'] * 0.50
        if row['Service Category'] == 'Lash Lifts': return row['revenue_after_vat'] * 0.40
        return 0
    df_transactions['Commission'] = df_transactions.apply(calculate_commission, axis=1)

    df_products['DATE'] = pd.to_datetime(df_products['DATE'], format='%d/%m/%Y', errors='coerce')
    df_products['PRICE/UNIT'] = pd.to_numeric(df_products['PRICE/UNIT'], errors='coerce').fillna(0)
    df_products['UNITS'] = pd.to_numeric(df_products['UNITS'], errors='coerce').fillna(0)
    df_products['Product Cost'] = df_products['PRICE/UNIT'] * df_products['UNITS']
    df_products['Month'] = df_products['DATE'].dt.month
    df_products['Year'] = df_products['DATE'].dt.year

    monthly_commission = df_transactions.groupby(['Artist', 'Year', 'Month'])['Commission'].sum().reset_index()
    monthly_product_cost = df_products.groupby(['ARTIST', 'Year', 'Month'])['Product Cost'].sum().reset_index()
    merged_monthly_data = pd.merge(monthly_commission, monthly_product_cost, left_on=['Artist', 'Year', 'Month'], right_on=['ARTIST', 'Year', 'Month'], how='left')
    merged_monthly_data = merged_monthly_data.drop('ARTIST', axis=1).fillna(0)
    merged_monthly_data['Net Salary'] = merged_monthly_data['Commission'] - merged_monthly_data['Product Cost']
    
    complaints = df_transactions.groupby(['Artist', 'Year', 'Month'])['Complaint'].sum().reset_index()
    redos = df_transactions[df_transactions['Service Type'] == 'Redo'].groupby(['Artist', 'Year', 'Month']).size().reset_index(name='Number of Redos')
    unique_clients = df_transactions.groupby(['Artist', 'Year', 'Month'])['Client Name'].nunique().reset_index().rename(columns={'Client Name': 'Unique Clients'})
    monthly_performance_data = pd.merge(complaints, redos, on=['Artist', 'Year', 'Month'], how='outer')
    monthly_performance_data = pd.merge(monthly_performance_data, unique_clients, on=['Artist', 'Year', 'Month'], how='outer').fillna(0)

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
    
    commission_by_service = df_transactions.groupby(['Artist', 'Year', 'Month', 'Service Type'])['Commission'].sum().reset_index()

    merged_monthly_data['MonthYear'] = pd.to_datetime(merged_monthly_data[['Year', 'Month']].assign(day=1))
    monthly_performance_data['MonthYear'] = pd.to_datetime(monthly_performance_data[['Year', 'Month']].assign(day=1))
    commission_by_service['MonthYear'] = pd.to_datetime(commission_by_service[['Year', 'Month']].assign(day=1))

    return merged_monthly_data.to_json(date_format='iso', orient='split'), \
           monthly_performance_data.to_json(date_format='iso', orient='split'), \
           retention_data.to_json(date_format='iso', orient='split'), \
           df_transactions.to_json(date_format='iso', orient='split'), \
           commission_by_service.to_json(date_format='iso', orient='split')

# --- Step 3: The App Factory ---
def create_dash_app():
    app = dash.Dash(__name__, external_stylesheets=[APP_THEME])
    initial_metrics, initial_performance, initial_retention, initial_transactions, initial_commission_by_service = load_and_process_data()
    app.layout = dbc.Container(fluid=True, className="app-container", children=[
        dcc.Store(id='metrics-data-store', data=initial_metrics),
        dcc.Store(id='performance-data-store', data=initial_performance),
        dcc.Store(id='retention-data-store', data=initial_retention),
        dcc.Store(id='transactions-data-store', data=initial_transactions),
        dcc.Store(id='commission-by-service-store', data=initial_commission_by_service),
        dbc.Row(dbc.Col(html.H1("Lash Studio Performance Dashboard"), width=12, className="text-center my-4")),
        dbc.Row([
            dbc.Col(html.H5(id='live-clock', className="text-start"), width=6),
            dbc.Col(dbc.Button("Refresh Data", id="refresh-button", n_clicks=0, color="primary", className="float-end"), width=6)
        ], className="mb-4"),
        dcc.Interval(id='interval-clock', interval=1000),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Select Artist", className="card-title"), dcc.Dropdown(id='artist-dropdown', value='All')])]), md=4, className="mb-4"),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("Start Month", className="card-title"), dcc.Dropdown(id='start-month-dropdown')])]), md=4, className="mb-4"),
            dbc.Col(dbc.Card([dbc.CardBody([html.H5("End Month", className="card-title"), dcc.Dropdown(id='end-month-dropdown')])]), md=4, className="mb-4"),
        ]),
        dbc.Tabs(id="dashboard-tabs", active_tab="tab-financials", children=[
            dbc.Tab(label="Financial Overview", tab_id="tab-financials"),
            dbc.Tab(label="Performance & Quality", tab_id="tab-performance"),
            dbc.Tab(label="Client Retention", tab_id="tab-retention"),
            dbc.Tab(label="Transaction Log", tab_id="tab-log"),
        ]),
        html.Div(id="tab-content", className="mt-4")
    ])
    register_callbacks(app)
    return app

def register_callbacks(app):
    @app.callback(
        Output('metrics-data-store', 'data', allow_duplicate=True),
        Output('performance-data-store', 'data', allow_duplicate=True),
        Output('retention-data-store', 'data', allow_duplicate=True),
        Output('transactions-data-store', 'data', allow_duplicate=True),
        Output('commission-by-service-store', 'data', allow_duplicate=True),
        Input('refresh-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def refresh_data_and_store(n_clicks):
        if n_clicks > 0: return load_and_process_data()
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output('artist-dropdown', 'options'),
        Output('start-month-dropdown', 'options'),
        Output('end-month-dropdown', 'options'),
        Output('start-month-dropdown', 'value'),
        Output('end-month-dropdown', 'value'),
        Input('metrics-data-store', 'data')
    )
    def update_controls(metrics_json):
        if not metrics_json or pd.read_json(metrics_json, orient='split').empty: return dash.no_update
        df = pd.read_json(metrics_json, orient='split')
        df['MonthYear'] = pd.to_datetime(df['MonthYear'])
        unique_artists = sorted(df['Artist'].unique())
        artist_options = [{'label': 'All Artists', 'value': 'All'}] + [{'label': artist, 'value': artist} for artist in unique_artists]
        available_months = sorted(df['MonthYear'].dt.to_period('M').astype(str).unique())
        month_options = [{'label': month, 'value': month} for month in available_months]
        start_month, end_month = (available_months[0], available_months[-1]) if available_months else (None, None)
        return artist_options, month_options, month_options, start_month, end_month

    @app.callback(
        Output('tab-content', 'children'),
        Input('dashboard-tabs', 'active_tab'),
        Input('artist-dropdown', 'value'),
        Input('start-month-dropdown', 'value'),
        Input('end-month-dropdown', 'value'),
        State('metrics-data-store', 'data'),
        State('performance-data-store', 'data'),
        State('retention-data-store', 'data'),
        State('transactions-data-store', 'data'),
        State('commission-by-service-store', 'data')
    )
    def render_tab_content(active_tab, selected_artist, start_month, end_month, 
                           metrics_json, performance_json, retention_json, transactions_json, commission_by_service_json):
        if not all([active_tab, selected_artist, start_month, end_month, metrics_json, performance_json, retention_json, transactions_json, commission_by_service_json]): return "" 
        
        merged_monthly_data = pd.read_json(metrics_json, orient='split')
        monthly_performance_data = pd.read_json(performance_json, orient='split')
        retention_data = pd.read_json(retention_json, orient='split')
        commission_by_service = pd.read_json(commission_by_service_json, orient='split')

        merged_monthly_data['MonthYear'] = pd.to_datetime(merged_monthly_data['MonthYear'])
        monthly_performance_data['MonthYear'] = pd.to_datetime(monthly_performance_data['MonthYear'])
        retention_data['MonthYear'] = pd.to_datetime(retention_data['MonthYear'])
        commission_by_service['MonthYear'] = pd.to_datetime(commission_by_service['MonthYear'])
        
        start_date = pd.to_datetime(start_month)
        end_date = pd.to_datetime(end_month) + pd.offsets.MonthEnd(0)

        metrics_by_date = merged_monthly_data[(merged_monthly_data['MonthYear'] >= start_date) & (merged_monthly_data['MonthYear'] <= end_date)]
        performance_by_date = monthly_performance_data[(monthly_performance_data['MonthYear'] >= start_date) & (monthly_performance_data['MonthYear'] <= end_date)]
        retention_by_date = retention_data[(retention_data['MonthYear'] >= start_date) & (retention_data['MonthYear'] <= end_date)]
        commission_by_service_by_date = commission_by_service[(commission_by_service['MonthYear'] >= start_date) & (commission_by_service['MonthYear'] <= end_date)]
        
        if selected_artist == 'All':
            title_name = "All Artists"
            metrics_display_df = metrics_by_date.groupby('MonthYear').agg(Commission=('Commission', 'sum'), **{'Net Salary': ('Net Salary', 'sum')}).reset_index()
            performance_display_df = performance_by_date.groupby('MonthYear').agg(Complaint=('Complaint', 'sum'), **{'Number of Redos': ('Number of Redos', 'sum')}, **{'Unique Clients': ('Unique Clients', 'sum')}).reset_index()
            retention_display_df = retention_by_date.groupby('MonthYear').agg(**{'Retention Rate': ('Retention Rate', 'mean')}).reset_index()
            commission_by_service_display_df = commission_by_service_by_date.groupby(['MonthYear', 'Service Type'])['Commission'].sum().reset_index()
        else:
            title_name = selected_artist
            metrics_display_df = metrics_by_date[metrics_by_date['Artist'] == selected_artist]
            performance_display_df = performance_by_date[performance_by_date['Artist'] == selected_artist]
            retention_display_df = retention_by_date[retention_by_date['Artist'] == selected_artist]
            commission_by_service_display_df = commission_by_service_by_date[commission_by_service_by_date['Artist'] == selected_artist]

        color_arg = {'color': 'Artist'} if 'Artist' in metrics_display_df.columns else {}

        if active_tab == 'tab-financials':
            if metrics_display_df.empty: return dbc.Alert(f"No financial data available.", color="info")
            total_commission = int(metrics_display_df['Commission'].sum())
            total_net_salary = int(metrics_display_df['Net Salary'].sum())
            fig_commission = px.line(metrics_display_df, x='MonthYear', y='Commission', title=f'Total Commission Trend for {title_name}', markers=True, **color_arg)
            fig_commission_breakdown = px.bar(commission_by_service_display_df, x='MonthYear', y='Commission', color='Service Type', title=f'Commission Breakdown by Service for {title_name}')
            return html.Div([
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4([html.Span("Ksh ", className="kpi-currency"), f"{total_commission:,.0f}"]), html.P("Total Commission", className="kpi-title")])), width=12, sm=6, md=6, className="mb-4"),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4([html.Span("Ksh ", className="kpi-currency"), f"{total_net_salary:,.0f}"]), html.P("Total Net Salary", className="kpi-title")])), width=12, sm=6, md=6, className="mb-4"),
                ], className="text-center"),
                dbc.Row([
                    dbc.Col(dbc.Card(dcc.Graph(figure=fig_commission)), md=6),
                    dbc.Col(dbc.Card(dcc.Graph(figure=fig_commission_breakdown)), md=6),
                ]),
                dbc.Accordion([dbc.AccordionItem(dbc.Table.from_dataframe(metrics_display_df.round(2), striped=True, bordered=True, hover=True), title="View Detailed Monthly Breakdown")], start_collapsed=True, className="mt-4")
            ])

        elif active_tab == 'tab-performance':
            if performance_display_df.empty: return dbc.Alert(f"No performance data available.", color="info")
            total_complaints = int(performance_display_df['Complaint'].sum())
            total_redos = int(performance_display_df['Number of Redos'].sum())
            total_clients = int(performance_display_df['Unique Clients'].sum())
            fig_performance = go.Figure()
            fig_performance.add_trace(go.Bar(x=performance_display_df['MonthYear'], y=performance_display_df['Unique Clients'], name='Unique Clients Served', marker_color='lightblue'))
            fig_performance.add_trace(go.Scatter(x=performance_display_df['MonthYear'], y=performance_display_df['Complaint'], name='Complaints', mode='lines+markers', yaxis='y2', line=dict(color='orange')))
            fig_performance.add_trace(go.Scatter(x=performance_display_df['MonthYear'], y=performance_display_df['Number of Redos'], name='Redos', mode='lines+markers', yaxis='y2', line=dict(color='red')))
            fig_performance.update_layout(title=f'Performance Scorecard for {title_name}', xaxis_title='Month', yaxis=dict(title='Unique Clients Served'), yaxis2=dict(title='Count', overlaying='y', side='right'), legend=dict(x=0, y=1.1, orientation='h'), template='plotly_white')
            return html.Div([
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{total_clients}"), html.P("Unique Clients Served", className="kpi-title")])), width=12, md=4, className="mb-4"),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{total_complaints}"), html.P("Total Complaints", className="kpi-title")])), width=12, md=4, className="mb-4"),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{total_redos}"), html.P("Total Redos", className="kpi-title")])), width=12, md=4, className="mb-4"),
                ], className="text-center"),
                dbc.Row([dbc.Col(dbc.Card(dcc.Graph(figure=fig_performance)), width=12)])
            ])

        elif active_tab == 'tab-retention':
            if retention_display_df.empty: return dbc.Alert(f"No retention data available.", color="info")
            avg_retention = float(retention_display_df['Retention Rate'].mean()) if not retention_display_df.empty else 0.0
            fig_retention = px.line(retention_display_df, x='MonthYear', y='Retention Rate', title=f'Client Retention Rate for {title_name}', markers=True, **color_arg)
            return html.Div([
                dbc.Row([dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{avg_retention:.1f}%"), html.P("Avg. Retention Rate", className="kpi-title")])), width=12, md=6, className="mx-auto mb-4")], className="text-center"),
                dbc.Row([dbc.Col(dbc.Card(dcc.Graph(figure=fig_retention)), width=12)]),
                dbc.Accordion([dbc.AccordionItem(dbc.Table.from_dataframe(retention_display_df.round(2), striped=True, bordered=True, hover=True), title="Retention Cohort Data")], start_collapsed=True, className="mt-4")
            ])
        
        elif active_tab == 'tab-log':
            if not transactions_json: return dbc.Alert("Transaction data not available.", color="warning")
            df_trans = pd.read_json(transactions_json, orient='split')
            df_trans['Date of Visit'] = pd.to_datetime(df_trans['Date of Visit'])
            filtered_df = df_trans[(df_trans['Date of Visit'] >= start_date) & (df_trans['Date of Visit'] <= end_date)]
            if selected_artist != 'All':
                filtered_df = filtered_df[filtered_df['Artist'] == selected_artist]
            filtered_df['Date of Visit'] = filtered_df['Date of Visit'].dt.strftime('%Y-%m-%d')
            display_columns = ['Date of Visit', 'Client Name', 'Artist', 'Service Type', 'Amount Paid', 'Code','Complaint']
            return dash_table.DataTable(
                id='transaction-table',
                columns=[{"name": i, "id": i} for i in display_columns],
                data=filtered_df[display_columns].to_dict('records'),
                filter_action="native", sort_action="native", page_action="native", page_size=20,
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
            )

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
    app.run(debug=True)