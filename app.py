import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()
transactions_url = os.environ['TRANSACTIONS_URL']
products_url = os.environ['PRODUCTS_URL']

# Define all functions here
def load_data(transactions_url, products_url):
    """
    Loads transaction and product data from specified URLs into pandas DataFrames.

    Args:
        transactions_url (str): URL of the transactions data CSV.
        products_url (str): URL of the products data CSV.

    Returns:
        tuple: A tuple containing the transactions DataFrame (df_transactions)
               and the products DataFrame (df_products).
    """
    # Read the CSV data from the provided URLs into pandas DataFrames
    df_transactions = pd.read_csv(transactions_url)
    df_products = pd.read_csv(products_url)
    return df_transactions, df_products

def clean_transactions_data(df_transactions):
    """
    Cleans and processes the transactions DataFrame.

    Includes cleaning 'Service Type', converting 'Amount Paid' and 'Date of Visit'
    to appropriate types, extracting month and year, calculating revenue after VAT,
    categorizing service types, and calculating artist commission.

    Args:
        df_transactions (pd.DataFrame): The raw transactions DataFrame.

    Returns:
        pd.DataFrame: The cleaned and processed transactions DataFrame.
    """
    # Remove leading/trailing spaces from 'Service Type' column for consistent formatting
    df_transactions['Service Type'] = df_transactions['Service Type'].str.strip()

    # Define a mapping for replacing misspelled or inconsistent service type values
    replace_map = {
        'Hybrid': 'hybrid',
        'hybrid  ': 'hybrid',
        'classic ': 'classic',
        'Russian vol.': 'russian volume',
        'Russian volume': 'russian volume',
        'removal+hybrid': 'hybrid+removal',
        'Lash lift': 'lash lift',
        'Russ refill': 'russian refill',
        'Russian volume refill': 'russian volume refill',
        'hybrid +tint': 'hybrid',
        'Russian refill': 'russian refill',
        'lash lift & Tint': 'lash lift & tint',
        'lash lift& tint': 'lash lift & tint',
        'classci refill': 'classic refill',
        'classic +removal': 'classic+removal',
        'Russian Volume': 'russian volume',
        'hybrid + brow tint': 'hybrid + brow tint',
        'hybrd refill': 'hybrid refill',
        'russian+removal': 'russian volume+removal',
        'hybrd': 'hybrid',
        'clasiic': 'classic',
        'hybrid   ': 'hybrid',
        'classic infill': 'classic refill',
        'hybrid + removal': 'hybrid+removal',
        'classic + removal': 'classic+removal',
        'mega refill': 'mega volume refill',
        'mega + removal': 'mega volume+removal',
    }
    # Apply the replacement map to the 'Service Type' column
    df_transactions['Service Type'] = df_transactions['Service Type'].replace(replace_map)

    # Convert 'Amount Paid' column to numeric, coercing errors to NaN
    df_transactions['Amount Paid'] = pd.to_numeric(df_transactions['Amount Paid'], errors='coerce')

    # Convert 'Date of Visit' column to datetime objects using the specified format
    df_transactions['Date of Visit'] = pd.to_datetime(df_transactions['Date of Visit'], format='%d/%m/%Y')

    # Extract month and year into new columns from 'Date of Visit'
    df_transactions['Month'] = df_transactions['Date of Visit'].dt.month
    df_transactions['Year'] = df_transactions['Date of Visit'].dt.year

    # Calculate revenue after deducting 16% VAT
    df_transactions['revenue_after_vat'] = df_transactions['Amount Paid'] * (1 - 0.16)

    # Define a helper function to categorize service types for commission calculation
    def categorize_service(service_type):
        extensions_removals = ['classic', 'hybrid', 'russian volume', 'refill', 'classic refill', 'hybrid refill', 'russian volume refill', 'classic+removal', 'hybrid+removal', 'removal', 'russian volume+removal', 'mega volume', 'mega volume refill', 'mega volume+removal', 'redo']
        lash_lifts = ['lash lift', 'lash lift & tint']
        if service_type in extensions_removals:
            return 'Extensions/Removals'
        elif service_type in lash_lifts:
            return 'Lash Lifts'
        else:
            return None

    # Apply the categorization function to create a new 'Service Category' column
    df_transactions['Service Category'] = df_transactions['Service Type'].apply(categorize_service)

    # Define a helper function to calculate commission based on service category and revenue after VAT
    def calculate_commission(row):
        if row['Service Category'] == 'Extensions/Removals':
            return row['revenue_after_vat'] * 0.50 # 50% commission for extensions/removals
        elif row['Service Category'] == 'Lash Lifts':
            return row['revenue_after_vat'] * 0.40 # 40% commission for lash lifts
        else:
            return 0 # 0 commission for uncategorized services

    # Apply the commission calculation function to create a new 'Commission' column
    df_transactions['Commission'] = df_transactions.apply(calculate_commission, axis=1)

    return df_transactions

def clean_products_data(df_products):
    """
    Cleans and processes the products DataFrame.

    Includes converting 'DATE' to datetime, calculating product cost,
    and extracting month and year.

    Args:
        df_products (pd.DataFrame): The raw products DataFrame.

    Returns:
        pd.DataFrame: The cleaned and processed products DataFrame.
    """
    # Convert 'DATE' column to datetime objects using the specified format
    df_products['DATE'] = pd.to_datetime(df_products['DATE'], format='%d/%m/%Y')

    # Calculate the total product cost for each transaction by multiplying price per unit and units
    df_products['Product Cost'] = df_products['PRICE/UNIT'] * df_products['UNITS']

    # Extract month and year into new columns from the 'DATE' column
    df_products['Month'] = df_products['DATE'].dt.month
    df_products['Year'] = df_products['DATE'].dt.year

    return df_products


# Load and clean data
df_transactions, df_products = load_data(transactions_url, products_url)
df_transactions = clean_transactions_data(df_transactions)
df_products = clean_products_data(df_products)

def calculate_monthly_artist_metrics(df_transactions, df_products):
    """
    Calculates monthly commission, product cost, and net salary per artist.

    Args:
        df_transactions (pd.DataFrame): The cleaned transactions DataFrame.
        df_products (pd.DataFrame): The cleaned products DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing monthly commission, product cost,
                      and net salary per artist.
    """
    # Group transactions by Artist, Year, and Month to sum the calculated commission
    monthly_artist_commission = df_transactions.groupby(['Artist', 'Year', 'Month'])['Commission'].sum()
    # Group products by Artist, Year, and Month to sum the calculated product cost
    monthly_artist_product_cost = df_products.groupby(['ARTIST', 'Year', 'Month'])['Product Cost'].sum()

    # Convert the grouped Series to DataFrames and reset their indices for merging
    monthly_artist_commission_df = monthly_artist_commission.reset_index()
    monthly_artist_product_cost_df = monthly_artist_product_cost.reset_index()

    merged_monthly_data = pd.merge(
        monthly_artist_commission_df,
        monthly_artist_product_cost_df,
        left_on=['Artist', 'Year', 'Month'],
        right_on=['ARTIST', 'Year', 'Month'],
        how='left' # Use a left merge to keep all commission data and add product costs where available
    )

    # Drop the redundant 'ARTIST' column resulting from the merge
    merged_monthly_data = merged_monthly_data.drop('ARTIST', axis=1)

    # Fill any missing product costs (artists with no product sales in a month) with 0
    merged_monthly_data['Product Cost'] = merged_monthly_data['Product Cost'].fillna(0)

    # Calculate the Net Salary by subtracting the Product Cost from the Commission
    merged_monthly_data['Net Salary'] = merged_monthly_data['Commission'] - merged_monthly_data['Product Cost']

    return merged_monthly_data

def calculate_client_retention(df_transactions, start_date, end_date, retention_period_months):
    """
    Calculates client retention rate per artist based on visits within a date range.

    Args:
        df_transactions (pd.DataFrame): The cleaned transactions DataFrame.
        start_date (datetime): The start date for the analysis.
        end_date (datetime): The end date for the analysis.
        retention_period_months (int): The number of months for the retention period.

    Returns:
        pd.DataFrame: A DataFrame with client retention data per artist and month.
    """
    # Sort the transactions data by Client Name and Date of Visit to easily identify first visits
    df_transactions_sorted = df_transactions.sort_values(by=['Client Name', 'Date of Visit'])

    # Identify the earliest visit date for each unique client
    first_visit_dates = df_transactions_sorted.groupby('Client Name')['Date of Visit'].min().reset_index()
    # Rename the column for clarity
    first_visit_dates.rename(columns={'Date of Visit': 'First Visit Date'}, inplace=True)

    # Merge the first visit dates back into the original transactions data
    df_transactions_with_first_visit = pd.merge(
        df_transactions_sorted,
        first_visit_dates,
        on='Client Name',
        how='left' # Use a left merge to keep all transaction data and add the first visit date
    )

    # Filter the transactions data to include only visits within the specified date range
    filtered_transactions_for_retention = df_transactions_with_first_visit[
        (df_transactions_with_first_visit['Date of Visit'] >= start_date) &
        (df_transactions_with_first_visit['Date of Visit'] <= end_date)
    ].copy() # Use .copy() to avoid potential SettingWithCopyWarning

    # Calculate the month (as a Period object) of the first visit within the filtered data
    filtered_transactions_for_retention['First Visit Month'] = filtered_transactions_for_retention['First Visit Date'].dt.to_period('M')

    # Calculate the month (as a Period object) of each visit within the filtered data
    filtered_transactions_for_retention['Visit Month'] = filtered_transactions_for_retention['Date of Visit'].dt.to_period('M')

    # Filter to include only the first visit for each client within the date range to determine the initial cohort size
    first_visits_only_filtered = filtered_transactions_for_retention[
        filtered_transactions_for_retention['Visit Month'] == filtered_transactions_for_retention['First Visit Month']
    ].drop_duplicates(subset=['Client Name', 'Artist', 'First Visit Month']) # Drop duplicates to count each client only once per artist and first visit month

    # Group by Artist and First Visit Month to calculate the number of unique clients in each cohort
    cohort_size_filtered = first_visits_only_filtered.groupby(['Artist', 'First Visit Month'])['Client Name'].nunique().reset_index(name='Cohort Size')

    # Identify clients who returned within the specified retention period after their first visit and within the date range
    returned_within_period_filtered = filtered_transactions_for_retention[
        (filtered_transactions_for_retention['Visit Month'] > filtered_transactions_for_retention['First Visit Month']) & # Visit is after the first visit
        (filtered_transactions_for_retention['Visit Month'] <= (filtered_transactions_for_retention['First Visit Month'] + retention_period_months)) # Visit is within the retention period
    ].drop_duplicates(subset=['Client Name', 'Artist', 'First Visit Month']) # Drop duplicates to count each returning client only once per artist and first visit month

    # Group by Artist and First Visit Month to count the number of returning clients within the period
    returning_clients_within_period_filtered = returned_within_period_filtered.groupby(['Artist', 'First Visit Month'])['Client Name'].nunique().reset_index(name=f'Returning Clients within {retention_period_months} Months')

    # Merge the cohort size and returning clients dataframes
    retention_data_period_filtered = pd.merge(
        cohort_size_filtered,
        returning_clients_within_period_filtered,
        on=['Artist', 'First Visit Month'],
        how='left' # Use a left merge to keep all cohorts and add returning client counts where available
    ).fillna(0) # Fill missing returning client counts with 0

    # Calculate the retention rate as a percentage
    retention_data_period_filtered[f'Retention Rate ({retention_period_months} Months)'] = (retention_data_period_filtered[f'Returning Clients within {retention_period_months} Months'] / retention_data_period_filtered['Cohort Size']) * 100

    return retention_data_period_filtered

def calculate_monthly_complaints(df_transactions):
    """
    Calculates the monthly number of complaints per artist.

    Args:
        df_transactions (pd.DataFrame): The cleaned transactions DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the monthly number of complaints per artist.
    """
    # Group transactions by Artist, Year, and Month to count the number of complaints
    monthly_artist_complaints = df_transactions.groupby(['Artist', 'Year', 'Month'])['Complaint'].count().reset_index()
    return monthly_artist_complaints

def calculate_monthly_redos(df_transactions):
    """
    Calculates the monthly number of redos per artist.

    Args:
        df_transactions (pd.DataFrame): The cleaned transactions DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the monthly number of redos per artist.
    """
    # Filter transactions for 'redo' service type
    df_redos = df_transactions[df_transactions['Service Type'] == 'redo'].copy()
    # Group redos by Artist, Year, and Month to count the number of redos
    monthly_artist_redos = df_redos.groupby(['Artist', 'Year', 'Month']).size().reset_index(name='Number of Redos')
    return monthly_artist_redos


# Calculate metrics
merged_monthly_data = calculate_monthly_artist_metrics(df_transactions, df_products)
monthly_artist_complaints = calculate_monthly_complaints(df_transactions)
monthly_artist_redos = calculate_monthly_redos(df_transactions)

# Prepare data for Dash
merged_monthly_data['MonthYear'] = pd.to_datetime(merged_monthly_data['Year'].astype(str) + '-' + merged_monthly_data['Month'].astype(str))
monthly_artist_complaints['MonthYear'] = pd.to_datetime(monthly_artist_complaints['Year'].astype(str) + '-' + monthly_artist_complaints['Month'].astype(str))
monthly_artist_redos['MonthYear'] = pd.to_datetime(monthly_artist_redos['Year'].astype(str) + '-' + monthly_artist_redos['Month'].astype(str))

# Merge monthly complaints and redos data
monthly_complaints_redos = pd.merge(
    monthly_artist_complaints,
    monthly_artist_redos,
    on=['Artist', 'Year', 'Month', 'MonthYear'],
    how='outer' # Use outer merge to keep all months from both dataframes
).fillna(0) # Fill any missing values (months with no complaints or redos) with 0


# Define retention period for client retention calculation
retention_period_months = 3
start_date = pd.to_datetime('2023-01-01')
end_date = datetime.now()

# Calculate client retention
retention_data_period_filtered = calculate_client_retention(
    df_transactions, start_date, end_date, retention_period_months
)
retention_data_period_filtered['MonthYear'] = retention_data_period_filtered['First Visit Month'].dt.to_timestamp()


# Get unique artists for the dropdown
unique_artists = sorted(merged_monthly_data['Artist'].unique())
artist_options = [{'label': artist, 'value': artist} for artist in unique_artists]

# Get unique months and years from the merged data and sort them for the date range picker
all_months = sorted(merged_monthly_data['MonthYear'].unique())
min_date_allowed = all_months[0]
max_date_allowed = all_months[-1]

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.layout = html.Div([
        html.H1("Artist Performance Dashboard", style={'textAlign': 'center', 'color': '#503D36'}),

        html.Div([
            html.Label("Select Artist(s):", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='artist-dropdown',
                options=artist_options,
                value=unique_artists,  # Default to all artists
                multi=True,
                style={'width': '50%', 'padding': '3px', 'fontSize': '18px', 'textAlignLast': 'center'}
            )
        ], style={'display': 'flex', 'marginBottom': '20px'}),

        html.Div([
            html.Label("Select Date Range:", style={'marginRight': '10px'}),
            dcc.DatePickerRange(
                id='date-range-picker',
                min_date_allowed=min_date_allowed,
                max_date_allowed=max_date_allowed,
                start_date=min_date_allowed,
                end_date=max_date_allowed,
                display_format='YYYY-MM',
                month_format='YYYY-MM',
                updatemode='bothdates',
                style={'fontSize': '18px'}
            )
        ], style={'display': 'flex', 'marginBottom': '20px'}),

        html.H2("Monthly Metrics", style={'marginTop': '30px', 'marginBottom': '10px', 'color': '#503D36'}),
        html.Div(id='monthly-commission-net-salary-table', style={'border': '1px solid #ddd', 'padding': '10px', 'marginBottom': '20px'}),
        html.Div(id='monthly-retention-table', style={'border': '1px solid #ddd', 'padding': '10px', 'marginBottom': '20px'}),
        html.Div(id='monthly-complaints-redos-table', style={'border': '1px solid #ddd', 'padding': '10px', 'marginBottom': '20px'}),
        html.Div(id='total-complaints-output', style={'marginBottom': '20px', 'fontSize': '18px', 'fontWeight': 'bold'}),
        html.Div(id='total-redos-output', style={'marginBottom': '20px', 'fontSize': '18px', 'fontWeight': 'bold'}),


        html.H2("Monthly Trends", style={'marginTop': '30px', 'marginBottom': '10px', 'color': '#503D36'}),
        html.Div(id='commission-trend-graph', style={'border': '1px solid #ddd', 'padding': '10px', 'marginBottom': '20px'}),
        html.Div(id='net-salary-trend-graph', style={'border': '1px solid #ddd', 'padding': '10px', 'marginBottom': '20px'}),
        html.Div(id='retention-trend-graph', style={'border': '1px solid #ddd', 'padding': '10px', 'marginBottom': '20px'}),
        html.Div(id='complaints-redos-trend-graph', style={'border': '1px solid #ddd', 'padding': '10px', 'marginBottom': '20px'}),
    ])

@app.callback(
        Output('monthly-commission-net-salary-table', 'children'),
        Output('monthly-retention-table', 'children'),
        Output('monthly-complaints-redos-table', 'children'),
        Output('total-complaints-output', 'children'),
        Output('total-redos-output', 'children'),
        Output('commission-trend-graph', 'children'),
        Output('net-salary-trend-graph', 'children'),
        Output('retention-trend-graph', 'children'),
        Output('complaints-redos-trend-graph', 'children'),
        Input('artist-dropdown', 'value'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date')
    )
def update_dashboard(selected_artists, start_date, end_date):
        if not selected_artists or not start_date or not end_date:
            return html.Div("Please select a date range and at least one artist."), "", "", "", "", "", "", "", ""

        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        # Filter dataframes for the selected date range and artists
        filtered_monthly_metrics = merged_monthly_data[
            (merged_monthly_data['MonthYear'] >= start_date_dt) &
            (merged_monthly_data['MonthYear'] <= end_date_dt) &
            (merged_monthly_data['Artist'].isin(selected_artists))
        ]
        filtered_retention_data = retention_data_period_filtered[
            (retention_data_period_filtered['MonthYear'] >= start_date_dt) &
            (retention_data_period_filtered['MonthYear'] <= end_date_dt) &
            (retention_data_period_filtered['Artist'].isin(selected_artists))
        ]
        filtered_complaints_redos_data = monthly_complaints_redos[
             (monthly_complaints_redos['MonthYear'] >= start_date_dt) &
             (monthly_complaints_redos['MonthYear'] <= end_date_dt) &
             (monthly_complaints_redos['Artist'].isin(selected_artists))
        ]


        # Calculate total complaints and redos
        total_complaints = filtered_complaints_redos_data['Complaint'].sum()
        total_redos = filtered_complaints_redos_data['Number of Redos'].sum()


        # Generate HTML table for monthly commission and net salary
        commission_net_salary_table = html.Div([
            html.H3(f"Monthly Commission and Net Salary ({start_date} to {end_date}) (Filtered by Artist)"),
            html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in ['Artist', 'Year', 'Month', 'Commission', 'Net Salary']])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(row['Artist']),
                        html.Td(row['Year']),
                        html.Td(row['Month']),
                        html.Td(f"{row['Commission']:.2f}"),
                        html.Td(f"{row['Net Salary']:.2f}")
                    ]) for index, row in filtered_monthly_metrics.iterrows()
                ])
            ])
        ])

        # Generate HTML table for monthly retention
        retention_table = html.Div([
            html.H3(f"Client Retention Rate ({start_date} to {end_date}) Cohort (Filtered by Artist)"),
            html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in ['Artist', 'First Visit Month', 'Cohort Size', f'Returning Clients within {retention_period_months} Months', f'Retention Rate ({retention_period_months} Months)']])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(row['Artist']),
                        html.Td(row['First Visit Month'].strftime('%Y-%m')),
                        html.Td(row['Cohort Size']),
                        html.Td(row[f'Returning Clients within {retention_period_months} Months']),
                        html.Td(f"{row[f'Retention Rate ({retention_period_months} Months)']:.2f}%")
                    ]) for index, row in filtered_retention_data.iterrows()
                ])
            ])
        ])

        # Generate HTML table for monthly complaints and redos
        complaints_redos_table = html.Div([
             html.H3(f"Complaints and Redos ({start_date} to {end_date}) (Filtered by Artist)"),
             html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in ['Artist', 'Year', 'Month', 'Number of Complaints', 'Number of Redos']])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(row['Artist']),
                        html.Td(row['Year']),
                        html.Td(row['Month']),
                        html.Td(row['Complaint']),
                        html.Td(row['Number of Redos'])
                    ]) for index, row in filtered_complaints_redos_data.iterrows()
                ])
            ])
        ])

        # Generate HTML for total complaints and redos
        total_complaints_output = html.Div(f"Total Complaints within the selected period: {total_complaints}")
        total_redos_output = html.Div(f"Total Redos within the selected period: {total_redos}")


        # Create trend graphs
        commission_trend_graph = dcc.Graph(
            figure=px.line(
                filtered_monthly_metrics,
                x='MonthYear',
                y='Commission',
                color='Artist',
                title='Monthly Commission Trend per Artist'
            ).update_layout(xaxis_title='Month and Year', yaxis_title='Commission', hovermode='x unified')
        )

        net_salary_trend_graph = dcc.Graph(
            figure=px.line(
                filtered_monthly_metrics,
                x='MonthYear',
                y='Net Salary',
                color='Artist',
                title='Monthly Net Salary Trend per Artist'
            ).update_layout(xaxis_title='Month and Year', yaxis_title='Net Salary', hovermode='x unified')
        )

        retention_trend_graph = dcc.Graph(
            figure=px.line(
                filtered_retention_data,
                x='MonthYear',
                y=f'Retention Rate ({retention_period_months} Months)',
                color='Artist',
                title=f'Monthly Retention Rate Trend per Artist ({retention_period_months} Months)'
            ).update_layout(xaxis_title='Month and Year', yaxis_title=f'Retention Rate ({retention_period_months} Months)', hovermode='x unified')
        )

        complaints_redos_trend_graph = dcc.Graph(
            figure=px.line(
                filtered_complaints_redos_data,
                x='MonthYear',
                y=['Complaint', 'Number of Redos'],
                color='Artist',
                title='Monthly Complaints and Redos Trend per Artist'
            ).update_layout(xaxis_title='Month and Year', yaxis_title='Count', hovermode='x unified')
        )


        return commission_net_salary_table, retention_table, complaints_redos_table, total_complaints_output, total_redos_output, commission_trend_graph, net_salary_trend_graph, retention_trend_graph, complaints_redos_trend_graph

if __name__ == '__main__':
	app.run_server(debug=True)
