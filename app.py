import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# model
from model import prediction
from sklearn.svm import SVR

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])
server = app.server

# Define the layout components

# Navigation component
item1 = html.Div(
    [
        html.P("Welcome to the Stock Dash App!", className="start"),

        html.Div([
            # stock code input
            dcc.Input(id='stock-code', type='text', placeholder='Enter stock code'),
            html.Button('Submit', id='submit-button')
        ], className="stock-input"),

        html.Div([
            # Date range picker input
            dcc.DatePickerRange(
                id='date-range', start_date=dt(2020, 1, 1).date(), end_date=dt.now().date(), className='date-input')
        ]),
        html.Div([
            # Stock price button
            html.Button('Get Stock Price', id='stock-price-button'),

            # Indicators button
            html.Button('Get Indicators', id='indicators-button'),

            # Number of days of forecast input
            dcc.Input(id='forecast-days', type='number', placeholder='Enter number of days'),

            # Forecast button
            html.Button('Get Forecast', id='forecast-button')
        ], className="selectors")
    ],
    className="nav"
)

# Content component
item2 = html.Div(
    [
        html.Div(
            [
                html.Img(id='logo', className='logo'),
                html.H1(id='company-name', className='company-name')
            ],
            className="header"),
        html.Div(id="description"),
        html.Div([], id="graphs-content"),
        html.Div([], id="main-content"),
        html.Div([], id="forecast-content")
    ],
    className="content"
)

# Set the layout
app.layout = html.Div(className='container', children=[item1, item2])

# Callbacks

# Callback to update the data based on the submitted stock code
@app.callback(
    [
        Output("description", "children"),
        Output("logo", "src"),
        Output("company-name", "children"),
        Output("stock-price-button", "n_clicks"),
        Output("indicators-button", "n_clicks"),
        Output("forecast-button", "n_clicks")
    ],
    [Input("submit-button", "n_clicks")],
    [State("stock-code", "value")]
)
def update_data(n, val):

    val = val.strip() if val else None

    if n is None:
        return None, None, None, None, None, None
    else:
        if val is None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info
            if 'logo_url' not in inf:
                return None, None, None, None, None, None
            else:
                name = inf['longName']
                logo_url = inf['logo_url']
                description = inf['longBusinessSummary']
                return description, logo_url, name, None, None, None


# Callback for displaying stock price graphs
@app.callback(
    [Output("graphs-content", "children")],
    [
        Input("stock-price-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def stock_price(n, start_date, end_date, val):
    if n is None or val is None:
        return [""]

    # Ensure start_date and end_date are provided
    if start_date is None or end_date is None:
        return ["Please select a valid date range."]

    # Download the data for the specified date range
    df = yf.download(val, start=start_date, end=end_date)

    if df.empty:
        return ["No data available for the selected date range."]

    df.reset_index(inplace=True)

    # Ensure 'Close' and 'Open' columns are 1D
    close_prices = df['Close'].values.flatten()  # Convert to 1D
    open_prices = df['Open'].values.flatten()    # Convert to 1D
    dates = df['Date']

    fig = px.line(x=dates, y=[close_prices, open_prices], 
                   labels={'x': 'Date', 'y': 'Price'}, 
                   title="Closing and Opening Price vs Date",
                   template='plotly')
    
    fig.update_traces(name='Close', selector=dict(name='Close'))
    fig.add_scatter(x=dates, y=open_prices, mode='lines', name='Open')

    return [dcc.Graph(figure=fig)]

# Callback for displaying indicators
@app.callback(
    [Output("main-content", "children")],
    [
        Input("indicators-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def indicators(n, start_date, end_date, val):
    if n is None or val is None:
        return [""]

    if start_date is None or end_date is None:
        return ["Please select a valid date range."]

    df_more = yf.download(val, start=start_date, end=end_date)

    if df_more.empty:
        return ["No data available for the selected date range."]

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]


def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig

# Callback for displaying forecast
@app.callback(
    Output("forecast-content", "children"),
    [Input("forecast-button", "n_clicks")],
    [State("forecast-days", "value"),
     State("stock-code", "value")]
)
def forecast(n, n_days, val):
    if n is None or val is None:
        return [""]

    if n_days is None or n_days <= 0:
        return [html.Div("Please enter a valid number of days for forecasting.")]

    try:
        
        fig = prediction(val, int(n_days) + 1) 
        return [dcc.Graph(figure=fig)]
    except ValueError as e:
        return [html.Div(str(e))]


if __name__ == '__main__':
    app.run_server(debug=True)
