import dash
import plotly.express as px
import pandas as pd
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from joblib import load

# Pomysł!!
# Funkcja która bierze znormalizowane i przygotowane dane, wyświetla wszystkie kolumny
# w multiselect liście i pozwala na podstawie zaznaczoyuch kolumn przetrenować model
# następnie zwraca informacje takie jak błąd średnio kwadratowy oraz jakiś wykres

df = pd.read_csv('diamonds_cleared.csv')
y_pred_sfs = load('y_pred_sfs.joblib')
model_sfs = load('model_sfs.joblib')
X_test_sfs = pd.read_csv('X_test_sfs.csv')
y_test_sfs = pd.read_csv('y_test_sfs.csv')
X_train_sfs = pd.read_csv('X_train_sfs.csv')
continuous_columns = ['carat', 'x dimension', 'y dimension', 'z dimension', 'depth', 'table', 'price']
categorical_columns = ['clarity', 'color', 'cut']

app = dash.Dash(__name__)
def update_residuals_plot():
    # Dokonaj przewidywań modelu
    y_pred_sfs = model_sfs.predict(X_test_sfs)
    # Oblicz reszty
    residuals = y_test_sfs['price'] - y_pred_sfs  # Użyj nazwy kolumny 'price', jeśli taka istnieje

    # Stwórz wykres
    fig = px.scatter(x=y_pred_sfs, y=residuals, labels={'x': 'Przewidywane wartości', 'y': 'Reszty'})
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        bargap=0.2
    )
    return fig
def update_feature_importance_plot():
    feature_importance = pd.DataFrame(model_sfs.coef_, index=X_train_sfs.columns, columns=['importance'])
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Stwórz wykres
    fig = px.bar(feature_importance, x='importance', y=feature_importance.index, orientation='h')
    fig.update_layout(title='Wizualizacja ważności cech dla modelu selekcji postępującej',
                      xaxis_title='Ważność cechy',
                      yaxis_title='Nazwa cechy',
                      bargap=0.2
    )
    return fig
def update_predicted_vs_actual_plot():
    # Stwórz wykres
    fig = px.scatter(x=y_test_sfs['price'], y=y_pred_sfs)  # Użyj nazwy kolumny 'price'
    fig.add_scatter(x=[y_test_sfs['price'].min(), y_test_sfs['price'].max()],
                    y=[y_test_sfs['price'].min(), y_test_sfs['price'].max()],
                    mode='lines', line=dict(color='red', dash='dash'),
                    name='Linia Idealna')
    fig.update_layout(title='Porównanie rzeczywistych i przewidywanych wartości dla modelu selekcji postępującej',
                      xaxis_title='Rzeczywiste wartości',
                      yaxis_title='Przewidywane wartości',
                      bargap=0.2)

    return fig


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------
app.layout = html.Div([
    html.Div(className='header', children=[
        html.H1("Dashboard - s21415")
    ]),

    html.Div(className='dropdown', children=[
        html.H2("Rozkład zmiennych"),
        dcc.Dropdown(
            id='continuous-dropdown',
            options=[{'label': col, 'value': col} for col in continuous_columns],
            value=continuous_columns[0]
        )
    ]),

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
    ),
    html.Div([
        dcc.Graph(figure=update_residuals_plot(), id='residuals-plot'),
        dcc.Graph(figure=update_feature_importance_plot(), id='feature-importance-plot'),
        dcc.Graph(figure=update_predicted_vs_actual_plot(), id='predicted-vs-actual-plot')

    ])
], style={'fontFamily': 'Arial, sans-serif'})


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


@app.callback(
    Output('price-dependency-scatter', 'figure'),
    [Input('price-dependency-dropdown', 'value')]
)
def update_price_dependency_scatter(selected_column):
    fig = px.scatter(df, x=selected_column, y='price')
    return fig


@app.callback(
    Output('category-countplot', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_category_countplot(selected_column):
    fig = px.histogram(df, x=selected_column)
    fig.update_layout(bargap=0.2)
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
