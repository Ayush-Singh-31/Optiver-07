import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import glob

PLOTLY_TEMPLATE = "plotly_white"
DBC_THEME = dbc.themes.LUX
DATA_DIR = "/Users/ayush/Documents/University/Year 03/Sem 01/DATA3888/Optiver-07/Dash WebApp/Data"
GENERAL_CSV_PATH = "/Users/ayush/Documents/University/Year 03/Sem 01/DATA3888/Optiver-07/Dash WebApp/Data/General.csv"
MODEL_FOLDERS = ["WLS", "Random Forest", "LSTM", "Transformer"]

try:
    df_general = pd.read_csv(GENERAL_CSV_PATH)
    df_general['stock_id'] = df_general['stock_id'].astype(str)
except FileNotFoundError:
    print(f"ERROR: '{GENERAL_CSV_PATH}' not found. Please ensure it exists or re-run to generate dummy data.")
    df_general = pd.DataFrame(columns=['stock_id', 'model_name', 'mse', 'qlike', 'r^2'])

app = dash.Dash(__name__, external_stylesheets=[DBC_THEME], suppress_callback_exceptions=True)
app.title = "Advanced Volatility Model Analyzer"

app.layout = dbc.Container(fluid=True, className="py-3", children=[
    dbc.Row(dbc.Col(html.H1("Advanced Volatility Model Analyzer", className="text-center text-primary mb-4"))),

    dbc.Card(className="mb-4 shadow-sm", body=True, children=[
        html.H3("Overall Model Performance (from general.csv)", className="card-title text-info mb-3"),
        dbc.Row([
            dbc.Col(md=3, children=[
                dbc.Label("Select Stock ID (Overall):"),
                dcc.Dropdown(
                    id='overall-stock-id-dropdown',
                    options=[{'label': i, 'value': i} for i in sorted(df_general['stock_id'].unique())],
                    value=sorted(df_general['stock_id'].unique())[0] if df_general['stock_id'].nunique() > 0 else None,
                    clearable=False,
                )
            ]),
        ]),
        dbc.Row([
            dbc.Col(md=4, className="mt-3", children=dcc.Loading(dcc.Graph(id='overall-mse-plot'))),
            dbc.Col(md=4, className="mt-3", children=dcc.Loading(dcc.Graph(id='overall-qlike-plot'))),
            dbc.Col(md=4, className="mt-3", children=dcc.Loading(dcc.Graph(id='overall-r2-plot'))),
        ]),
    ]),

    dbc.Card(className="mb-4 shadow-sm", body=True, children=[
        html.H3("Detailed Stock/Time ID Analysis", className="card-title text-success mb-3"),
        dbc.Row([
            dbc.Col(md=3, children=[
                dbc.Label("Select Model Type:"),
                dcc.Dropdown(id='detail-model-type-dropdown', options=[{'label': m, 'value': m} for m in MODEL_FOLDERS], clearable=False, value=MODEL_FOLDERS[0]),
            ]),
            dbc.Col(md=3, children=[
                dbc.Label("Select Stock ID File:"),
                dcc.Dropdown(id='detail-stock-id-file-dropdown', clearable=False),
            ]),
            dbc.Col(md=3, children=[
                dbc.Label("Select Time ID (from file):"), 
                dcc.Dropdown(id='detail-time-id-dropdown', clearable=False),
            ]),
            dbc.Col(md=3, className="align-self-end", children=[
                dbc.Button("Load & Analyze Segment", id='load-analyze-button', color="primary", className="w-100")
            ]),
        ]),
        html.Div(id='detailed-analysis-plots-container', className="mt-3", children=[
            html.P("Select model, stock ID file, time ID, and click 'Load & Analyze' to see detailed plots.", className="text-muted")
        ])
    ]),
])

@app.callback(
    Output('overall-mse-plot', 'figure'),
    Output('overall-qlike-plot', 'figure'),
    Output('overall-r2-plot', 'figure'),
    Input('overall-stock-id-dropdown', 'value')
)
def update_overall_plots(selected_overall_stock_id):
    if not selected_overall_stock_id or df_general.empty:
        empty_fig = go.Figure().update_layout(template=PLOTLY_TEMPLATE, title_text="No Data")
        return empty_fig, empty_fig, empty_fig

    filtered_df = df_general[df_general['stock_id'] == selected_overall_stock_id]

    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(template=PLOTLY_TEMPLATE, title_text=f"No Data for Stock {selected_overall_stock_id}")
        return empty_fig, empty_fig, empty_fig

    fig_mse = px.bar(filtered_df, x='model_name', y='mse', color='model_name',
                     title=f'MSE Comparison for Stock {selected_overall_stock_id}', template=PLOTLY_TEMPLATE,
                     labels={'model_name': 'Model', 'mse': 'Mean Squared Error'})
    fig_qlike = px.bar(filtered_df, x='model_name', y='qlike', color='model_name',
                       title=f'QLIKE Comparison for Stock {selected_overall_stock_id}', template=PLOTLY_TEMPLATE,
                       labels={'model_name': 'Model', 'qlike': 'QLIKE'})
    fig_r2 = px.bar(filtered_df, x='model_name', y='r^2', color='model_name',
                    title=f'R² Score Comparison for Stock {selected_overall_stock_id}', template=PLOTLY_TEMPLATE,
                    labels={'model_name': 'Model', 'r^2': 'R² Score'})

    fig_mse.update_layout(showlegend=False)
    fig_qlike.update_layout(showlegend=False)
    fig_r2.update_layout(showlegend=False)

    return fig_mse, fig_qlike, fig_r2

@app.callback(
    Output('detail-stock-id-file-dropdown', 'options'),
    Output('detail-stock-id-file-dropdown', 'value'),  
    Input('detail-model-type-dropdown', 'value')
)
def set_stock_id_file_options(selected_model_type):
    if not selected_model_type:
        return [], None

    model_data_path = os.path.join(DATA_DIR, selected_model_type)
    if not os.path.exists(model_data_path):
        return [], None

    csv_files = sorted([f for f in os.listdir(model_data_path) if f.endswith('.csv') and f[:-4].isdigit()])
    options = [{'label': f, 'value': f} for f in csv_files]
    value = csv_files[0] if csv_files else None
    return options, value

@app.callback(
    Output('detail-time-id-dropdown', 'options'),
    Output('detail-time-id-dropdown', 'value'),   
    Input('detail-model-type-dropdown', 'value'),
    Input('detail-stock-id-file-dropdown', 'value')
)
def set_detail_time_id_options(selected_model_type, selected_stock_id_filename): 
    if not selected_model_type or not selected_stock_id_filename:
        return [], None

    file_path = os.path.join(DATA_DIR, selected_model_type, selected_stock_id_filename)
    if not os.path.exists(file_path):
        return [], None

    try:
        temp_df = pd.read_csv(file_path)
        temp_df['time_id'] = temp_df['time_id'].astype(str) 
        time_ids_from_file = sorted(temp_df['time_id'].unique())
        options = [{'label': t_id, 'value': t_id} for t_id in time_ids_from_file]
        value = time_ids_from_file[0] if time_ids_from_file else None
        return options, value
    except Exception as e:
        print(f"Error reading {file_path} for time IDs: {e}")
        return [], None

@app.callback(

    Output('detailed-analysis-plots-container', 'children'),
    Input('load-analyze-button', 'n_clicks'),
    State('detail-model-type-dropdown', 'value'),
    State('detail-stock-id-file-dropdown', 'value'),
    State('detail-time-id-dropdown', 'value'),      
    prevent_initial_call=True
)
def load_and_display_detailed_data(n_clicks, model_type, selected_stock_id_filename, selected_time_id_from_file):
    if not n_clicks or not model_type or not selected_stock_id_filename or not selected_time_id_from_file:
        return html.P("Please make all selections and click 'Load & Analyze'.", className="text-warning")

    file_path = os.path.join(DATA_DIR, model_type, selected_stock_id_filename)

    if not os.path.exists(file_path):
        return html.P(f"Error: File not found - {file_path}", className="text-danger")

    try:
        df_full_detailed = pd.read_csv(file_path)

        df_full_detailed['stock_id'] = df_full_detailed['stock_id'].astype(str)
        df_full_detailed['time_id'] = df_full_detailed['time_id'].astype(str)

        df_selected_segment = df_full_detailed[
            (df_full_detailed['time_id'] == selected_time_id_from_file) &
            (df_full_detailed['stock_id'] == selected_stock_id_filename.replace('.csv', ''))
        ].copy()

        if df_selected_segment.empty:
            return html.P(f"No data found for Stock: {selected_stock_id_filename.replace('.csv', '')}, Time ID: {selected_time_id_from_file} in {selected_stock_id_filename}.", className="text-warning")

        df_selected_segment['error'] = df_selected_segment['true_vol'] - df_selected_segment['pred_vol']

        stock_display_name = selected_stock_id_filename.replace('.csv', '')
        segment_metrics = df_selected_segment[['mse', 'qlike', 'r^2']].iloc[0]

        metrics_display = html.Div([
            html.H5(f"Metrics for {model_type} - Stock: {stock_display_name} - Time ID: {selected_time_id_from_file}", className="text-muted"),
            html.P(f"QLIKE: {segment_metrics['qlike']:.6f} | R²: {segment_metrics['r^2']:.6f}", className="small")
        ], className="mb-3")

        fig_scatter_detail = px.scatter(
            df_selected_segment, x='true_vol', y='pred_vol',
            labels={'true_vol': 'True Volatility', 'pred_vol': 'Predicted Volatility'},
            template=PLOTLY_TEMPLATE, marginal_y="histogram", marginal_x="histogram",
            title=f"Predicted vs. True Volatility (Stock: {stock_display_name}, Time: {selected_time_id_from_file})"
        )
        min_val = min(df_selected_segment['true_vol'].min(), df_selected_segment['pred_vol'].min()) if not df_selected_segment.empty else 0
        max_val = max(df_selected_segment['true_vol'].max(), df_selected_segment['pred_vol'].max()) if not df_selected_segment.empty else 1
        fig_scatter_detail.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color='Gray', dash='dash'))

        df_line_plot = df_selected_segment.reset_index().rename(columns={'index': 'observation_index'})
        fig_line_detail = go.Figure()
        fig_line_detail.add_trace(go.Scatter(x=df_line_plot['observation_index'], y=df_line_plot['true_vol'], mode='lines+markers', name='True Vol', line=dict(dash='dot')))
        fig_line_detail.add_trace(go.Scatter(x=df_line_plot['observation_index'], y=df_line_plot['pred_vol'], mode='lines+markers', name='Pred Vol'))
        fig_line_detail.update_layout(template=PLOTLY_TEMPLATE, xaxis_title='Observation Index', yaxis_title='Volatility', title=f"Volatility Over Time (Stock: {stock_display_name}, Time: {selected_time_id_from_file})")

        fig_hist_detail = px.histogram(
            df_selected_segment, x='error', template=PLOTLY_TEMPLATE,
            labels={'error': 'Prediction Error (True - Predicted)'},
            title=f"Prediction Error Distribution (Stock: {stock_display_name}, Time: {selected_time_id_from_file})", marginal="box"
        )
        fig_hist_detail.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")

        detailed_plots_layout = dbc.Container(fluid=True, children=[
            metrics_display,
            dbc.Row([
                dbc.Col(md=6, children=dcc.Graph(figure=fig_scatter_detail)),
                dbc.Col(md=6, children=dcc.Graph(figure=fig_line_detail)),
            ]),
            dbc.Row([
                dbc.Col(md=6, className="mt-3", children=dcc.Graph(figure=fig_hist_detail)),
                dbc.Col(md=6, className="mt-3") 
            ])
        ])

        return detailed_plots_layout

    except Exception as e:
        print(f"Error processing detailed data from {file_path}: {e}")
        return html.Div([
            html.P(f"An error occurred while loading or processing data from {file_path}.", className="text-danger"),
            html.Pre(str(e))
        ])

app.run(debug=True, port=8050)


