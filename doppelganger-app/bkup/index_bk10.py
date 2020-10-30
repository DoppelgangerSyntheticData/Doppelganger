import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import pathlib
import plotly.graph_objs as go
from plotly import tools
from dash.dependencies import Input, Output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from demo_utils import demo_callbacks, demo_explanation

global product_df
global dict_products


########################### Image model ####################################
DATA_PATH = pathlib.Path(__file__).parent.joinpath("data").resolve()
LOGFILE = "models/run_log.csv"
demo_mode = True

def div_graph(name):
    # Generates an html Div containing graph and control options for smoothing and display, given the name
    return html.Div(
        className="row",
        children=[
            html.Div(
                className="two columns",
                style={"padding-bottom": "5%"},
                children=[
                    html.Div(
                        [
                            html.Div(
                                className="graph-checkbox-smoothing",
                                children=["Smoothing:"],
                            ),
                            dcc.Checklist(
                                options=[
                                    {"label": " Training", "value": "train"},
                                    {"label": " Validation", "value": "val"},
                                ],
                                value=[],
                                id=f"checklist-smoothing-options-{name}",
                                className="checklist-smoothing",
                            ),
                        ],
                        style={"margin-top": "10px"},
                    ),
                    html.Div(
                        [
                            dcc.Slider(
                                min=0,
                                max=1,
                                step=0.05,
                                marks={i / 5: str(i / 5) for i in range(0, 6)},
                                value=0.6,
                                updatemode="drag",
                                id=f"slider-smoothing-{name}",
                            )
                        ],
                        style={"margin-bottom": "40px"},
                        className="slider-smoothing",
                    ),
                    html.Div(
                        [
                            html.P(
                                "Plot Display Mode:",
                                style={"font-weight": "bold", "margin-bottom": "0px"},
                                className="plot-display-text",
                            ),
                            html.Div(
                                [
                                    dcc.RadioItems(
                                        options=[
                                            {
                                                "label": " Overlapping",
                                                "value": "overlap",
                                            },
                                            {
                                                "label": " Separate (Vertical)",
                                                "value": "separate_vertical",
                                            },
                                            {
                                                "label": " Separate (Horizontal)",
                                                "value": "separate_horizontal",
                                            },
                                        ],
                                        value="overlap",
                                        id=f"radio-display-mode-{name}",
                                        labelStyle={"verticalAlign": "middle"},
                                        className="plot-display-radio-items",
                                    )
                                ],
                                className="radio-item-div",
                            ),
                            html.Div(id=f"div-current-{name}-value"),
                        ],
                        className="entropy-div",
                    ),
                ],
            ),
            html.Div(id=f"div-{name}-graph", className="ten columns"),
        ],
    )

########################### DATA SETS ####################################
df_le = pd.read_csv("data/life_expectancy_data.csv")
df_le_winz = pd.read_csv("data/life_expectancy_data_winz_index.csv")
###############################################################
app = dash.Dash(__name__,
                external_scripts=["/assets/index.css"],
                suppress_callback_exceptions=True)
app.title = 'Doppelgänger'
app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])
# ############ LAYOUTS ##################################################
page1Name = "Numeric Data Regression Analysis (WHO Dataset)"
page2Name = "Image Data Classification Analysis (MNIST Dataset)"
page1Url = "/" + page1Name.replace(" ", "-")
page2Url = "/" + page2Name.replace(" ", "-")

def  generate_table(dataframe, max_rows=10):
    return html.Table(
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in  range(min(len(dataframe), max_rows))],
        style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'middle'}
    )


pg0_table1 = generate_table(
    pd.DataFrame(
        {
            "Data Type": ["Numeric", "Image"],
            "Analysis Type": ["Regression", "Classification"],
            "Doppelgänger Product Page": [dcc.Link(page1Name, href=page1Url), dcc.Link(page2Name, href=page2Url)],
            "Dataset Source Page": [dcc.Link("Life-Expectancy", href="https://www.kaggle.com/kumarajarshi/life-expectancy-who?select=Life+Expectancy+Data.csv"),
                        dcc.Link("MNIST", href="http://yann.lecun.com/exdb/mnist/")],
            "Dataset Details": ["The Global Health Observatory (GHO) data repository under World Health Organization (WHO) keeps track of the health status of all countries. The dataset related to life expectancy, health factors for 193 countries has been collected from the same WHO data repository website and its corresponding economic data was collected from United Nation website.",
                        "The MNIST (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms."],
        }, columns=["Data Type","Doppelgänger Product Page","Analysis Type","Dataset Source Page","Dataset Details"]))


page_header = html.Div([
    html.Div([
        dcc.Link('Home', href='/'),
        dcc.Link('Featured', href='/'),
        dcc.Link('Product', href='/'),
        dcc.Link('About', href='/'),
        dcc.Link('Contact', href='/')
    ], className="topmenu-smalltext"),
    html.Img(src="/assets/favicon.ico", height="60pt", width="60pt"),
    " Doppelgänger ",
    html.Span("| Synthetic Data Generator", className="topmenu-span")
]
    , className="topmenu")

index_page = html.Div([
    page_header,
    html.H3("Catalog", className="content-header"),
    html.Br(),
    html.Div(className="catalog", children=[pg0_table1])
])

page_1_layout = html.Div([
    page_header,
    html.H1(page1Name, className="content-header"),
    dcc.Dropdown(
        id='page-1-dropdown',
        options=[{'label': i, 'value': i} for i in ['Data Buyer', 'Data Owner', 'Data Generator']],
        value='Data Buyer'
    ),
    html.Div(id='page-1-content'),
])

page_2_layout = html.Div([
    page_header,
    html.H1(page2Name, className="content-header"),
    dcc.Dropdown(
        id='page-2-radios',
        options=[{'label': i, 'value': i} for i in ['Data Buyer', 'Data Owner', 'Data Generator']],
        value='Data Buyer'
    ),
    html.Div(id='page-2-content'),
])
# -----
pg1_tab1 = html.Div([html.H1("Dataset Introduction",
                             style={'textAlign': 'center', 'width': '1024px', 'overflow-x': 'auto', 'white-space': 'nowrap'}),
                     dash_table.DataTable(
                          id='table',
                          columns=[{"name": i, "id": i} for i in df_le.columns],
                          data=df_le.iloc[0:5, :].to_dict("rows"),
                      ),
                     html.H1("Dataset after pre-processing",
                              style={'textAlign': 'center', 'width': '1024px', 'overflow-x': 'auto','white-space': 'nowrap'}),
                     dash_table.DataTable(
                          id='table',
                          columns=[{"name": i, "id": i} for i in df_le_winz.columns],
                          data=df_le_winz.iloc[0:5, :].to_dict("rows"),
                      ),
                     ])

pg1_tab2 = html.Div(className="container", children=[html.H1("XGBoost Validator Performance", style={"textAlign": "center"}),
                     html.H1("", style={'textAlign': "center", 'padding-top': 5}),
                     html.Img(src="/assets/cs1_comp_XGB.png", height="500pt", width="680pt"),
                     html.Div([
                          dcc.RadioItems(id="select-survival", value=str(1),
                                         labelStyle={'display': 'inline-block', 'padding': 10},
                                         options=[{'label': "Paid", 'value': str(1)},
                                                  {'label': "Free", 'value': str(0)}], )],
                          style={'textAlign': "center", }),
                     html.Div(
                          [html.Div([dcc.Graph(id="hist-graph", clear_on_unhover=True, )]), ]),
                     ])

pg1_tab3 = html.Div([html.H1("Random Forest Evaluator Performance", style={"textAlign": "center"}),
                     html.P(
              "R2 Score of RandomForestRegressor : 0.88, Root Mean Squared Error Score of RandomForestRegressor : 3.17 "),
                     html.Img(src="/assets/cs1_comp_RF.png", height="500pt", width="680pt"),
                     html.H2("Life Expectancy Actual vs Model Predicted Values", style={"textAlign": "left"}),
                     html.Img(src="/assets/cs1_comp_RFvsXGB.png", height="500pt", width="680pt"),
                     html.H3("Performance Metrics Regression Prediction", style={"textAlign": "left"}),
                     html.Img(src="/assets/cs1_comp_3stages.png", height="500pt", width="680pt"),
                     html.P("")], )

# ----
pg2_tab1 = html.Div(children=[html.Div(
                className="container",
                style={"padding": "35px 25px"},
                children=[
                    dcc.Store(id="storage-simulated-run", storage_type="memory"),
                    dcc.Interval(id="interval-simulated-step",interval=125,n_intervals=0),
                    html.Div(className="row",style={"margin": "8px 0px"},
                        children=[
                            html.Div(
                                className="twelve columns",
                                children=[
                                    html.Div(
                                        className="eight columns",
                                        children=[
                                            html.Div(
                                                dcc.Dropdown(
                                                    id="dropdown-demo-dataset",
                                                    options=[
                                                        {
                                                            "label": "MNIST",
                                                            "value": "mnist",
                                                        }
                                                    ],
                                                    value="mnist",
                                                    placeholder="dataset",
                                                    searchable=False,
                                                ),
                                                className="six columns dropdown-box-first",
                                            ),
                                            html.Div(
                                                dcc.Dropdown(
                                                    id="dropdown-simulation-model",
                                                    options=[
                                                        {
                                                            "label": "VAE",
                                                            "value": "cnn",
                                                        },
                                                    ],
                                                    value="cnn",
                                                    placeholder="Select Model",
                                                    searchable=False,
                                                ),
                                                className="six columns dropdown-box-second",
                                            ),
                                            html.Div(
                                                dcc.Dropdown(
                                                    id="dropdown-interval-control",
                                                    options=[
                                                        {
                                                            "label": "Model Training (Logs)",
                                                            "value": "regular",
                                                        }
                                                    ],
                                                    value="regular",
                                                    className="twelve columns dropdown-box-third",
                                                    clearable=False,
                                                    searchable=False,
                                                )
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="four columns",
                                        id="div-interval-control",
                                        children=[
                                            html.Div(
                                                id="div-total-step-count",
                                                className="twelve columns",
                                            ),
                                            html.Div(
                                                id="div-step-display",
                                                className="twelve columns",
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    dcc.Interval(id="interval-log-update", n_intervals=0),
                    dcc.Store(id="run-log-storage", storage_type="memory"),
                ],
            ),
            html.Div(className="container", children=[div_graph("accuracy")]),
            html.Div(
                className="container",
                style={"margin-bottom": "30px"},
                children=[div_graph("cross-entropy")],
            )])

pg2_table1 = generate_table(
    pd.DataFrame(
        {"": [
            html.H1("Dataset Introduction"),
            html.Div(className="container",children=["Load and show MNIST images."
                         , html.Br(),
                      "The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset."
                         , html.Br(), html.Img(src="/assets/cs2_comp_VAE_mnist.png"), html.Br()
                         , html.Br(), html.Img(src="/assets/MnistExamples.png"), html.Br()

                      ])
            , html.H1("VAE & Auto-Encoder")
            , html.Div(className="container",children=["Variational Autoencoders (VAEs) can be used to visualize high-dimensional data in a meaningful, lower-dimensional space. In this kernel, we will go over some details about autoencoding and autoencoders, especially VAEs, before constructing and training a deep VAE on the MNIST data from the Digit Recognizer competition. Autoencoder predictions are the compressed representations of the digits themselves"
                           , html.Br(), html.Img(src="/assets/cs2_VAE.png"), html.Br()
                           ])
            , html.H1("Model Construction")
            , html.Div(className="container",children=["Model construction for VAE uses convolution layers in encoder and decoder. VAE has three components - 1. An ENCODER that that learns the parameters (mean and variance) of the underlying latent distribution; 2. A means of SAMPLING from that distribution; and, 3. A DECODER that can turn the sample from #2 back into an image.",
                           html.Br(), html.Img(src="/assets/cs2_comp_VAE_train.png"), html.Br()
                           ])
            , html.H1("Model Training")
            , pg2_tab1
            , html.H1("Generated Data")
            , html.Div(className="container",children=[
                "Clustering of digits in the latent space We can make predictions on the validation set using the encoder network. This has the effect of translating the images from the 784-dimensional input space into the 2-dimensional latent space. When we color-code those translated data points according to their known digit class, we can see how the digits cluster together."
                , html.Br(), html.Img(src="/assets/cs2_comp_VAE_rep.png"), html.Br()
                ,
                "Reconstructing Digits: Autoencoder predictions are the compressed representations of the digits themselves"
                , html.Br(), html.Img(src="/assets/cs2_comp_VAE_compressed.png"), html.Br()
                ,
                "Another fun thing we can do is to use the decoder network to take a peak at what samples from the latent space look like as we change the latent variables. What we end up with is a smoothly varying space where each digit transforms into the others as we dial the latent variables up and down."
                , html.Br(), html.Img(src="/assets/cs2_comp_VAE_2.png"), html.Br()
            ])
        ],
        }, columns=[""]))

# -- CALLBACKS --

@app.callback(dash.dependencies.Output('page-1-content', 'children'),
              [dash.dependencies.Input('page-1-dropdown', 'value')])
def page_1_dropdown(value):
    return dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Data Generation', children=[pg1_tab1]),
        dcc.Tab(label='Data Validation', children=[pg1_tab2]),
        dcc.Tab(label='Data Evaluation', children=[pg1_tab3])
    ], className="container")

@app.callback(dash.dependencies.Output('page-2-content', 'children'),
              [dash.dependencies.Input('page-2-radios', 'value')])
def page_2_radios(value):
    return dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Data Generation', children=[pg2_table1]),
        dcc.Tab(label='Data Validation', children=[]),
        dcc.Tab(label='Data Evaluation', children=[])
    ])


@app.callback(Output('le_1', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph1(selected_dropdown):
    return go.Figure(data=go.Heatmap(
                    z=[[1, 20, 30],
                      [20, 1, 60],
                      [30, 60, 1]]))


############## Image Model Training #################################################

def update_graph(
    graph_id,
    graph_title,
    y_train_index,
    y_val_index,
    run_log_json,
    display_mode,
    checklist_smoothing_options,
    slider_smoothing,
    yaxis_title,
):
    """
    :param graph_id: ID for Dash callbacks
    :param graph_title: Displayed on layout
    :param y_train_index: name of column index for y train we want to retrieve
    :param y_val_index: name of column index for y val we want to retrieve
    :param run_log_json: the json file containing the data
    :param display_mode: 'separate' or 'overlap'
    :param checklist_smoothing_options: 'train' or 'val'
    :param slider_smoothing: value between 0 and 1, at interval of 0.05
    :return: dcc Graph object containing the updated figures
    """

    def smooth(scalars, weight=0.6):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    if run_log_json:
        layout = go.Layout(
            title=graph_title,
            margin=go.layout.Margin(l=50, r=50, b=50, t=50),
            yaxis={"title": yaxis_title},
        )

        run_log_df = pd.read_json(run_log_json, orient="split")

        step = run_log_df["step"]
        y_train = run_log_df[y_train_index]
        y_val = run_log_df[y_val_index]

        # Apply Smoothing if needed
        if "train" in checklist_smoothing_options:
            y_train = smooth(y_train, weight=slider_smoothing)

        if "val" in checklist_smoothing_options:
            y_val = smooth(y_val, weight=slider_smoothing)

        # line charts
        trace_train = go.Scatter(
            x=step,
            y=y_train,
            mode="lines",
            name="Training",
            line=dict(color="rgb(54, 218, 170)"),
            showlegend=False,
        )

        trace_val = go.Scatter(
            x=step,
            y=y_val,
            mode="lines",
            name="Validation",
            line=dict(color="rgb(246, 236, 145)"),
            showlegend=False,
        )

        if display_mode == "separate_vertical":
            figure = tools.make_subplots(rows=2, cols=1, print_grid=False)

            figure.append_trace(trace_train, 1, 1)
            figure.append_trace(trace_val, 2, 1)

            figure["layout"].update(
                title=layout.title,
                margin=layout.margin,
                scene={"domain": {"x": (0.0, 0.5), "y": (0.5, 1)}},
            )

            figure["layout"]["yaxis1"].update(title=yaxis_title)
            figure["layout"]["yaxis2"].update(title=yaxis_title)

        elif display_mode == "separate_horizontal":
            figure = tools.make_subplots(
                rows=1, cols=2, print_grid=False, shared_xaxes=True
            )

            figure.append_trace(trace_train, 1, 1)
            figure.append_trace(trace_val, 1, 2)

            figure["layout"].update(title=layout.title, margin=layout.margin)
            figure["layout"]["yaxis1"].update(title=yaxis_title)
            figure["layout"]["yaxis2"].update(title=yaxis_title)

        elif display_mode == "overlap":
            figure = go.Figure(data=[trace_train, trace_val], layout=layout)

        else:
            figure = None

        return dcc.Graph(figure=figure, id=graph_id)

    return dcc.Graph(id=graph_id)


demo_callbacks(app, demo_mode)


@app.callback(
    [Output("demo-explanation", "children"), Output("learn-more-button", "children")],
    [Input("learn-more-button", "n_clicks")],
)
def learn_more(n_clicks):
    if n_clicks is None:
        n_clicks = 0
    if (n_clicks % 2) == 1:
        n_clicks += 1
        return (
            html.Div(
                className="container",
                style={"margin-bottom": "30px"},
                children=[demo_explanation(demo_mode)],
            ),
            "Close",
        )

    n_clicks += 1
    return (html.Div(), "Learn More")


@app.callback(
    Output("interval-log-update", "interval"),
    [Input("dropdown-interval-control", "value")],
)
def update_interval_log_update(interval_rate):
    if interval_rate == "fast":
        return 500

    elif interval_rate == "regular":
        return 1000

    elif interval_rate == "slow":
        return 5 * 1000

    # Refreshes every 24 hours
    elif interval_rate == "no":
        return 24 * 60 * 60 * 1000


if not demo_mode:

    @app.callback(
        Output("run-log-storage", "data"), [Input("interval-log-update", "n_intervals")]
    )
    def get_run_log(_):
        names = [
            "step",
            "train accuracy",
            "val accuracy",
            "train cross entropy",
            "val cross entropy",
        ]

        try:
            run_log_df = pd.read_csv("DATA_PATH.joinpath(LOGFILE)", names=names)
            json = run_log_df.to_json(orient="split")
        except FileNotFoundError as error:
            print(error)
            print(
                "Please verify if the csv file generated by your model is placed in the correct directory."
            )
            return None

        return json


@app.callback(
    Output("div-step-display", "children"), [Input("run-log-storage", "data")]
)
def update_div_step_display(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")
        return html.H6(
            f"Step: {run_log_df['step'].iloc[-1]}",
            style={"margin-top": "3px", "float": "right"},
        )


@app.callback(
    Output("div-accuracy-graph", "children"),
    [
        Input("run-log-storage", "data"),
        Input("radio-display-mode-accuracy", "value"),
        Input("checklist-smoothing-options-accuracy", "value"),
        Input("slider-smoothing-accuracy", "value"),
    ],
)
def update_accuracy_graph(
    run_log_json, display_mode, checklist_smoothing_options, slider_smoothing
):
    graph = update_graph(
        "accuracy-graph",
        "Prediction Accuracy",
        "train accuracy",
        "val accuracy",
        run_log_json,
        display_mode,
        checklist_smoothing_options,
        slider_smoothing,
        "Accuracy",
    )

    try:
        if display_mode in ["separate_horizontal", "overlap"]:
            graph.figure.layout.yaxis["range"] = [0, 1.3]
            graph.figure.layout.yaxis2["range"] = [0, 1.3]
        else:
            graph.figure.layout.yaxis1["range"] = [0, 1.3]
            graph.figure.layout.yaxis2["range"] = [0, 1.3]

    except AttributeError:
        pass

    return [graph]


@app.callback(
    Output("div-cross-entropy-graph", "children"),
    [
        Input("run-log-storage", "data"),
        Input("radio-display-mode-cross-entropy", "value"),
        Input("checklist-smoothing-options-cross-entropy", "value"),
        Input("slider-smoothing-cross-entropy", "value"),
    ],
)
def update_cross_entropy_graph(
    run_log_json, display_mode, checklist_smoothing_options, slider_smoothing
):
    graph = update_graph(
        "cross-entropy-graph",
        "Cross Entropy Loss",
        "train cross entropy",
        "val cross entropy",
        run_log_json,
        display_mode,
        checklist_smoothing_options,
        slider_smoothing,
        "Loss",
    )
    return [graph]


@app.callback(
    Output("div-current-accuracy-value", "children"), [Input("run-log-storage", "data")]
)
def update_div_current_accuracy_value(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")
        return [
            html.P(
                "Current Accuracy:",
                style={
                    "font-weight": "bold",
                    "margin-top": "15px",
                    "margin-bottom": "0px",
                },
            ),
            html.Div(f"Training: {run_log_df['train accuracy'].iloc[-1]:.4f}"),
            html.Div(f"Validation: {run_log_df['val accuracy'].iloc[-1]:.4f}"),
        ]


@app.callback(
    Output("div-current-cross-entropy-value", "children"),
    [Input("run-log-storage", "data")],
)
def update_div_current_cross_entropy_value(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")
        return [
            html.P(
                "Current Loss:",
                style={
                    "font-weight": "bold",
                    "margin-top": "15px",
                    "margin-bottom": "0px",
                },
            ),
            html.Div(f"Training: {run_log_df['train cross entropy'].iloc[-1]:.4f}"),
            html.Div(f"Validation: {run_log_df['val cross entropy'].iloc[-1]:.4f}"),
        ]


############## MAIN #################################################
# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == page1Url:
        return page_1_layout
    elif pathname == page2Url:
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(debug=True)
