import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
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


# --- Data Loading and Cleaning Functions (No changes here) ---

def load_data(transactions_url, products_url):
    df_transactions = pd.read_csv(transactions_url)
    df_products = pd.read_csv(products_url)
    return df_transactions, df_products

def clean_transactions_data(df):
    df['Service Type'] = df['Service Type'].str.strip().str.lower()
    replace_map = {
        'hybrid': 'hybrid', 'hybrid  ': 'hybrid', 'classic ': 'classic', 'russian vol.': 'russian volume',
        'russian volume': 'russian volume', 'removal+hybrid': 'hybrid+removal', 'lash lift': 'lash lift',
        'russ refill': 'russian refill', 'russian volume refill': 'russian volume refill', 'hybrid +tint': 'hybrid',
        'russian refill': 'russian refill', 'lash lift & tint': 'lash lift & tint', 'lash lift& tint': 'lash lift & tint',
        'classci refill': 'classic refill', 'classic +removal': 'classic+removal', 'russian volume': 'russian volume',
        'hybrid + brow tint': 'hybrid + brow tint', 'hybrd refill': 'hybrid refill', 'hybrd': 'hybrid',
        'clasiic': 'classic', 'hybrid   ': 'hybrid', 'classic infill': 'classic refill', 'hybrid + removal': 'hybrid+removal',
        'classic + removal': 'classic+removal', 'mega refill': 'mega volume refill', 'mega + removal': 'mega volume+removal'
    }
    df['Service Type'] = df['Service Type'].replace(replace_map)
    df['Amount Paid'] = pd.to_numeric(df['Amount Paid'], errors='coerce')
    df['Date of Visit'] = pd.to_datetime(df['Date of Visit'], format='%d/%m/%Y')
    df['Month'] = df['Date of Visit'].dt.month
    df['Year'] = df['Date of Visit'].dt.year
    df['revenue_after_vat'] = df['Amount Paid'] * (1 - 0.16)

    def categorize_service(service_type):
        extensions = ['classic', 'hybrid', 'russian volume', 'refill', 'classic refill', 'hybrid refill', 'russian volume refill', 'classic+removal', 'hybrid+removal', 'removal', 'russian volume+removal', 'mega volume', 'mega volume refill', 'mega volume+removal', 'redo']
        lash_lifts = ['lash lift', 'lash lift & tint']
        if service_type in extensions: return 'Extensions/Removals'
        if service_type in lash_lifts: return 'Lash Lifts'
        return None
    df['Service Category'] = df['Service Type'].apply(categorize_service)

    def calculate_commission(row):
        if row['Service Category'] == 'Extensions/Removals': return row['revenue_after_vat'] * 0.50
        if row['Service Category'] == 'Lash Lifts': return row['revenue_after_vat'] * 0.40
        return 0
    df['Commission'] = df.apply(calculate_commission, axis=1)
    return df

def clean_products_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')
    df['Product Cost'] = df['PRICE/UNIT'] * df['UNITS']
    df['Month'] = df['DATE'].dt.month
    df['Year'] = df['DATE'].dt.year
    return df

# --- Initial Data Load and Processing ---
df_transactions_raw, df_products_raw = load_data(transactions_url, products_url)
df_transactions = clean_transactions_data(df_transactions_raw)
df_products = clean_products_data(df_products_raw)

# --- Metric Calculation Functions ---
def calculate_monthly_artist_metrics(df_trans, df_prod):
    monthly_commission = df_trans.groupby(['Artist', 'Year', 'Month'])['Commission'].sum().reset_index()
    monthly_product_cost = df_prod.groupby(['ARTIST', 'Year', 'Month'])['Product Cost'].sum().reset_index()
    merged_data = pd.merge(monthly_commission, monthly_product_cost, left_on=['Artist', 'Year', 'Month'], right_on=['ARTIST', 'Year', 'Month'], how='left')
    merged_data = merged_data.drop('ARTIST', axis=1).fillna(0)
    merged_data['Net Salary'] = merged_data['Commission'] - merged_data['Product Cost']
    return merged_data

def calculate_client_retention(df_trans, start_date, end_date, retention_months):
    df_sorted = df_trans.sort_values(by=['Client Name', 'Date of Visit'])
    first_visits = df_sorted.groupby('Client Name')['Date of Visit'].min().reset_index().rename(columns={'Date of Visit': 'First Visit Date'})
    df_with_first_visit = pd.merge(df_sorted, first_visits, on='Client Name', how='left')
    df_filtered = df_with_first_visit[(df_with_first_visit['Date of Visit'] >= start_date) & (df_with_first_visit['Date of Visit'] <= end_date)].copy()
    df_filtered['First Visit Month'] = df_filtered['First Visit Date'].dt.to_period('M')
    df_filtered['Visit Month'] = df_filtered['Date of Visit'].dt.to_period('M')
    first_visits_only = df_filtered[df_filtered['Visit Month'] == df_filtered['First Visit Month']].drop_duplicates(subset=['Client Name', 'Artist', 'First Visit Month'])
    cohort_size = first_visits_only.groupby(['Artist', 'First Visit Month'])['Client Name'].nunique().reset_index(name='Cohort Size')
    returned_clients = df_filtered[(df_filtered['Visit Month'] > df_filtered['First Visit Month']) & (df_filtered['Visit Month'] <= (df_filtered['First Visit Month'] + retention_months))].drop_duplicates(subset=['Client Name', 'Artist', 'First Visit Month'])
    returning_count = returned_clients.groupby(['Artist', 'First Visit Month'])['Client Name'].nunique().reset_index(name=f'Returning Clients')
    retention_data = pd.merge(cohort_size, returning_count, on=['Artist', 'First Visit Month'], how='left').fillna(0)
    retention_data[f'Retention Rate'] = (retention_data[f'Returning Clients'] / retention_data['Cohort Size']) * 100
    return retention_data

def calculate_monthly_complaints_redos(df_trans):
    complaints = df_trans.groupby(['Artist', 'Year', 'Month'])['Complaint'].count().reset_index()
    redos = df_trans[df_trans['Service Type'] == 'redo'].groupby(['Artist', 'Year', 'Month']).size().reset_index(name='Number of Redos')
    merged = pd.merge(complaints, redos, on=['Artist', 'Year', 'Month'], how='outer').fillna(0)
    return merged

# --- Prepare Data for Dash ---
merged_monthly_data = calculate_monthly_artist_metrics(df_transactions, df_products)
monthly_complaints_redos = calculate_monthly_complaints_redos(df_transactions)
retention_data = calculate_client_retention(df_transactions, pd.to_datetime('2023-01-01'), datetime.now(), 3)
merged_monthly_data['MonthYear'] = pd.to_datetime(merged_monthly_data[['Year', 'Month']].assign(day=1))
monthly_complaints_redos['MonthYear'] = pd.to_datetime(monthly_complaints_redos[['Year', 'Month']].assign(day=1))
retention_data['MonthYear'] = retention_data['First Visit Month'].dt.to_timestamp()

# --- Prepare Dropdown Options ---
unique_artists = sorted(merged_monthly_data['Artist'].unique())
artist_options = [{'label': 'All Artists', 'value': 'All'}] + [{'label': artist, 'value': artist} for artist in unique_artists]
min_date = merged_monthly_data['MonthYear'].min()
max_date = merged_monthly_data['MonthYear'].max()

# --- Initialize the Dash App ---
app = dash.Dash(__name__, external_stylesheets=[APP_THEME])
server = app.server

# --- App Layout ---
app.layout = dbc.Container(fluid=True, className="app-container", children=[
    dbc.Row(dbc.Col(html.H1("Artist Performance Dashboard"), width=12, className="text-center my-4")),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H5("Select Artist", className="card-title"),
            dcc.Dropdown(id='artist-dropdown', options=artist_options, value='All')
        ])]), md=6, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H5("Select Date Range", className="card-title"),
            dcc.DatePickerRange(
                id='date-range-picker',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date,
                display_format='MMM YYYY',
                className="w-100"
            )
        ])]), md=6, className="mb-4"),
    ]),
    # The Output of the callback will be rendered here
    html.Div(id='dashboard-content') 
])

# --- Main Callback ---
@app.callback(
    # The Output is the Div with the id 'dashboard-content'
    Output('dashboard-content', 'children'),
    # The Inputs that will trigger this function
    Input('artist-dropdown', 'value'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_dashboard(selected_artist, start_date_str, end_date_str):
    if not all([selected_artist, start_date_str, end_date_str]):
        return "" 

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Filter data by the selected date range first
    metrics_by_date = merged_monthly_data[(merged_monthly_data['MonthYear'] >= start_date) & (merged_monthly_data['MonthYear'] <= end_date)]
    complaints_by_date = monthly_complaints_redos[(monthly_complaints_redos['MonthYear'] >= start_date) & (monthly_complaints_redos['MonthYear'] <= end_date)]
    retention_by_date = retention_data[(retention_data['MonthYear'] >= start_date) & (retention_data['MonthYear'] <= end_date)]

    # Now, handle the artist selection
    if selected_artist == 'All':
        title_name = "All Artists (Studio Total)"
        metrics_df = metrics_by_date.groupby('MonthYear').agg({'Commission': 'sum', 'Net Salary': 'sum'}).reset_index()
        complaints_df = complaints_by_date.groupby('MonthYear').agg({'Complaint': 'sum', 'Number of Redos': 'sum'}).reset_index()
        retention_df = retention_by_date.groupby('MonthYear').agg({'Retention Rate': 'mean'}).reset_index()
    else:
        title_name = selected_artist
        metrics_df = metrics_by_date[metrics_by_date['Artist'] == selected_artist]
        complaints_df = complaints_by_date[complaints_by_date['Artist'] == selected_artist]
        retention_df = retention_by_date[retention_by_date['Artist'] == selected_artist]

    if metrics_df.empty:
        return dbc.Alert(f"No data available for {title_name} in the selected date range.", color="info", className="m-4")

    # --- KPIs (with explicit type casting) ---
total_commission = int(metrics_df['Commission'].sum())
total_net_salary = int(metrics_df['Net Salary'].sum())
total_complaints = int(complaints_df['Complaint'].sum())
total_redos = int(complaints_df['Number of Redos'].sum())

    # Create Figures
color_arg = {'color': 'Artist'} if 'Artist' in metrics_df.columns else {}
fig_commission = px.line(metrics_df, x='MonthYear', y='Commission', title=f'Commission Trend for {title_name}', markers=True, **color_arg)
fig_net_salary = px.line(metrics_df, x='MonthYear', y='Net Salary', title=f'Net Salary Trend for {title_name}', markers=True, **color_arg)
fig_retention = px.line(retention_df, x='MonthYear', y='Retention Rate', title=f'Client Retention Rate for {title_name}', markers=True, **color_arg)
fig_complaints = px.bar(complaints_df, x='MonthYear', y=['Complaint', 'Number of Redos'], title=f'Complaints & Redos for {title_name}', barmode='group')

    # Return the layout to be displayed
    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"Ksh {total_commission:,.0f}"), html.P("Total Commission")])]), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"Ksh {total_net_salary:,.0f}"), html.P("Total Net Salary")])]), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{int(total_complaints)}"), html.P("Total Complaints")])]), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{int(total_redos)}"), html.P("Total Redos")])]), md=3),
        ], className="text-center mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_commission)), md=6, className="mb-4"),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_net_salary)), md=6, className="mb-4"),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_retention)), md=6, className="mb-4"),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_complaints)), md=6, className="mb-4"),
        ]),
        dbc.Accordion([
            dbc.AccordionItem(dbc.Table.from_dataframe(metrics_df.round(2), striped=True, bordered=True, hover=True), title="Monthly Salary Data"),
            dbc.AccordionItem(dbc.Table.from_dataframe(retention_df.round(2), striped=True, bordered=True, hover=True), title="Client Retention Data"),
            dbc.AccordionItem(dbc.Table.from_dataframe(complaints_df.round(2), striped=True, bordered=True, hover=True), title="Complaints & Redos Data"),
        ], start_collapsed=True)
    ])


if __name__ == '__main__':
    app.run_server(debug=True)