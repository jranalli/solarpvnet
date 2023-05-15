import os

import pandas as pd
from dash import html, dcc, Dash  # Dash layout
from dash import Input, Output, no_update  # Callbacks
from plotly.express import imshow
from PIL import Image


basetest = "NY-Q"
testnames = ["CA-S", "CA-F", "FR-G", "FR-I", "DE-G", "NY-Q"]
typenames = ["precision", "recall", "iou_score"]
modelnames = ["CA-S", "CA-F", "FR-G", "FR-I", "DE-G", "NY-Q", "CMB-6"]
# Learning from https://thingsolver.com/dash-by-plotly/

def choose_images(test, n=10, model_choice='global', choose_by="iou_score"):
    if model_choice == 'global':
        # Choose all the relevant models, excluding combo and the model for that test data
        avg_indices = testnames.copy()
        avg_indices.pop(avg_indices.index(test))
    else:
        # Choose this one specific model, overriding the other consideration
        avg_indices = [model_choice]

    data_path = rf"D:\data\solardnn\results\resnet34_42_1\{test}_test\resnet34_42_v1_{test}_imgmetrics.xlsx"
    # img_path = rf"D:\data\solardnn\{test}\tiles\img"
    # mask_path = rf"D:\data\solardnn\{test}\tiles\mask"

    df = pd.read_excel(data_path, ["precision", "recall", "iou_score"], header=0, index_col=0)

    p = df['precision']
    r = df['recall']
    iou = df['iou_score']

    avg_df = pd.DataFrame({"precision": p[avg_indices].mean(axis=1),
                           "recall": r[avg_indices].mean(axis=1),
                           "iou_score": iou[avg_indices].mean(axis=1)}
                          , index=p.index)
    avg_df = avg_df.sort_values(by=choose_by)
    worst = avg_df.index[:n].values.tolist()
    best = avg_df.index[-n:].values.tolist()
    worst.reverse()
    best.reverse()
    return worst, best


def imagefig(test,  model='global', file="060195_34.png", alph_r=0.5, alph_b=0.5):
    img_path = rf"D:\data\solardnn\{test}\tiles\img"
    mask_path = rf"D:\data\solardnn\{test}\tiles\mask"
    im = Image.open(os.path.join(img_path, file)).convert('RGB')
    red = Image.new('RGB', im.size, (255, 0, 0))
    ms = Image.open(os.path.join(mask_path, file)).convert('L').point(lambda i: i*alph_r)
    im = Image.composite(red, im, ms).convert('RGB')

    if not model == 'global':
        pred_path = rf"D:\data\solardnn\{model}\predictions\{model}_resnet34_42_v1_predicting_{test}\pred_masks"
        blue = Image.new('RGB', im.size, (0, 0, 255))
        pr = Image.open(os.path.join(pred_path, file)).convert('L').point(lambda i: i*alph_b).resize(im.size)
        im = Image.composite(blue, im, pr).convert('RGB')

    fig = imshow(im)
    fig.update_layout(
        height=275,
        title=f"{file}",
        # minreducedheight=150,
        margin=dict(l=20, r=20, t=30, b=5, pad=0),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, automargin=True)
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )
    return fig


def build_app():
    # use bootstrap css
    external_stylesheets = [
        {
            'href': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
            'rel': 'stylesheet',
            'integrity': 'BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u',
            'crossorigin': 'anonymous'
        }
    ]
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    model_options = modelnames.copy()
    model_options.insert(0, "global")

    # Add each of the figures to a row
    children_list = [
        html.Div(
            id="header", className='row', style={"display": "block", "margin": "15px"},
            children=[
                html.H1(children='Best & Worst Imgages by IoU Score'),
                "Choose Test Data",
                dcc.Dropdown(testnames, testnames[0],  className='col-4', id='testdropdown', style={"width": "150px"}, clearable=False),
                "Choose Metric",
                dcc.Dropdown(typenames, typenames[0],  className='col-4', id='typedropdown', style={"width": "150px"}, clearable=False),
                "Choose Model",
                dcc.Dropdown(model_options, model_options[0], className='col-4', id='modeldropdown', style={"width": "150px"},  clearable=False),
                "Number of Imgs",
                dcc.Input(id="numimgs", type="number", value=12, min=1, max=100, step=1, className='col-4', style={"width": "150px"}, debounce=True),
            ]
        ),
        html.Div(
            className='row', style={"display": "block", "margin": "15px"},
            children=[
                html.H4(children='Best Images'),
            ]
        ),
        dcc.Loading(
            id="loading",
            children=[html.Div([html.Div(id="loading-output")])],
            type="circle",
            fullscreen=True
        ),
        html.Div(
            id="figbestrow", className='row', style={},
            children=[]
        ),
        html.Div(
            className='row', style={"display": "block", "margin": "15px"},
            children=[
                html.H4(children='Worst Images'),
            ]
        ),
        html.Div(
            id="figworstrow", className='row', style={},
            children=[]
        )
    ]
    app.layout = html.Div(children=children_list)

    @app.callback(
        Output("loading-output", "children"),
        Output("figbestrow", "children"),
        Output("figworstrow", "children"),
        Input("testdropdown", "value"),
        Input("typedropdown", "value"),
        Input("modeldropdown", "value"),
        Input("numimgs", "value"),
    )
    def callback(testvalue, typevalue, modelvalue, numvalue):
        worst, best = choose_images(testvalue, n=numvalue, model_choice=modelvalue, choose_by=typevalue)
        bestfigs = [imagefig(testvalue, modelvalue, name) for name in best]
        worstfigs = [imagefig(testvalue, modelvalue, name) for name in worst]

        return [], figs_to_divs(bestfigs), figs_to_divs(worstfigs)

    return app

def figs_to_divs(figs):
    figdivs = []
    for fig in figs:
        figdiv = html.Div(
            className='col-sm-6 col-md-3 col-lg-2',
            children=[
                dcc.Graph(figure=fig, config=dict(displayModeBar=False))
            ]
        )
        figdivs.append(figdiv)
    return figdivs



if __name__ == "__main__":
    # worst, best = choose_images(basetest, n=10)
    # figs = [imagefig(basetest, name) for name in best]
    app = build_app()
    app.run_server(debug=True)
