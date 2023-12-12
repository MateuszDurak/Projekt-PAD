import dash
import plotly.express as px
import pandas as pd
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output


df = pd.read_csv('diamonds_cleared.csv')
continuous_columns = ['carat', 'x dimension', 'y dimension', 'z dimension', 'depth', 'table', 'price']
categorical_columns = ['clarity', 'color', 'cut']

app = dash.Dash(__name__)

# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------
app.layout = html.Div([
    html.H1("Dashboard - s21415"),


    html.H2("Rozkład zmiennych"),
    dcc.Dropdown(
        id='continuous-dropdown',
        options=[{'label': col, 'value': col} for col in continuous_columns],
        value=continuous_columns[0]
    ),
    dcc.Graph(id='continuous-histogram'),


    html.H2("Zależności ceny od innych zmiennych"),
    dcc.Dropdown(
        id='price-dependency-dropdown',
        options=[{'label': col, 'value': col} for col in continuous_columns if col != 'price'],
        value=continuous_columns[0]
    ),
    dcc.Graph(id='price-dependency-scatter'),


    html.H2("Liczebność kategorii"),
    dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': col, 'value': col} for col in categorical_columns],
        value=categorical_columns[0]
    ),
    dcc.Graph(id='category-countplot'),
    html.H2("Podgląd Danych"),
    dash_table.DataTable(
        id='data-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=20,  # Liczba wierszy na stronę
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'lightgrey'
            }
        ],
        style_table={'height': '300px', 'overflowY': 'auto'}
    )
])


# --------------------------------------------------------------------------------------
# Callbacki
# --------------------------------------------------------------------------------------
@app.callback(
    Output('continuous-histogram', 'figure'),
    [Input('continuous-dropdown', 'value')]
)
def update_continuous_histogram(selected_column):
    fig = px.histogram(df, x=selected_column, marginal="box")
    fig.update_layout(bargap=0.2)
    return fig


# Callback dla zależności ceny od innych zmiennych
@app.callback(
    Output('price-dependency-scatter', 'figure'),
    [Input('price-dependency-dropdown', 'value')]
)
def update_price_dependency_scatter(selected_column):
    fig = px.scatter(df, x=selected_column, y='price')
    return fig


# Callback dla liczebności kategorii
@app.callback(
    Output('category-countplot', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_category_countplot(selected_column):
    fig = px.histogram(df, x=selected_column)
    return fig


def update_table(xaxis_column_name):
    table = html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(min(len(df), 10))
        ])
    ])
    return table


# Uruchomienie serwera
if __name__ == '__main__':
    app.run_server(debug=True)
