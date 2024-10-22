import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from datetime import date, timedelta
import plotly.graph_objs as go

def prediction(stock, n_days):
    # Load the data
    try:
        df = yf.download(stock, period='1mo')
        if df.empty:
            raise ValueError("No data found for the stock symbol provided.")
    except Exception as e:
        raise ValueError(f"Error downloading data: {str(e)}")

    df.reset_index(inplace=True)
    df['Day'] = df.index

    # Preprocess the data
    days = [[i] for i in range(len(df.Day))]
    X = days
    Y = df[['Close']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    # Train and select the model
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.001, 0.01, 0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 150, 1000],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
        },
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )
    
    y_train = y_train.values.ravel()
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"], max_iter=-1)

    # Train the model
    best_svr.fit(x_train, y_train)

    # Predict and visualize the results
    output_days = [[i + x_test[-1][0]] for i in range(1, n_days)]
    dates = [(date.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(x_test).flatten(), y=y_test.values.flatten(), mode='markers', name='data'))
    fig.add_trace(go.Scatter(x=np.array(x_test).flatten(), y=best_svr.predict(x_test), mode='lines+markers', name='test'))

    fig.add_trace(go.Scatter(x=dates, y=best_svr.predict(output_days), mode='lines+markers', name='data'))
    fig.update_layout(title=f"Predicted Close Price of next {n_days - 1} days", xaxis_title="Date", yaxis_title="Close Price")
    return fig
