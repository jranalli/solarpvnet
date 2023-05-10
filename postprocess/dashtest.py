import io
import base64

import os

from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import plotly.express as px

from PIL import Image

from sklearn.manifold import TSNE
import numpy as np
import pandas as pd


# https://dash.plotly.com/dash-core-components/tooltip

test = "FR-G"

data_path = rf"D:\data\solardnn\results\resnet34_42_1\{test}_test\resnet34_42_v1_{test}_imgmetrics.xlsx"
img_path = rf"D:\data\solardnn\{test}\tiles\img"
mask_path = rf"D:\data\solardnn\{test}\tiles\mask"



color = {"CA-S": 0, "CA-F": 1, "FR-G": 2, "FR-I": 3, "DE-G": 4, "NY-Q": 5, "CMB-6": 6}

def np_image_to_base64(im_path, msk_path, pre_path, im_name, alph_r=0.75, alph_b=0.5):
    im = Image.open(os.path.join(im_path, im_name)).convert('RGB')

    red = Image.new('RGB', im.size, (255, 0, 0))
    ms = Image.open(os.path.join(msk_path, im_name)).convert('L').point(lambda i: i*alph_r)
    im = Image.composite(red, im, ms).convert('RGB')

    blue = Image.new('RGB', im.size, (0, 0, 255))
    pr = Image.open(os.path.join(pre_path, im_name)).convert('L').point(lambda i: i*alph_b).resize(im.size)
    im = Image.composite(blue, im, pr).convert('RGB')

    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


df = pd.read_excel(data_path, ["precision","recall"], header=0, index_col=0)
p = df['precision']
r = df['recall']

ps = np.array([])
rs = np.array([])
ims = []
cs = []
for key in color.keys():
    ps = np.hstack([ps, p[key].values]) if ps.size else p[key].values
    rs = np.hstack([rs, r[key].values]) if rs.size else r[key].values
    ims.extend(list(p[key].index))
    ci = [key]*len(p[key].values)
    cs.extend(ci)

bdf = pd.DataFrame({"precision": ps, "recall": rs, "color": cs, "image": ims}, index=ims)

# fig = go.Figure(data=[
#     go.Scatter(
#         x=bdf['precision'],
#         y=bdf['recall'],
#         mode="markers",
#         marker=dict(
#             color=bdf['color'],
#             opacity=0.8,
#         )
#     )
# ])
# fig = px.scatter(
#         x=bdf['precision'],
#         y=bdf['recall'],
#         custom_data=np.array(bdf.index),
#         color=bdf['color'],
#         color_discrete_sequence=px.colors.qualitative.D3
#     )
fig = px.scatter(np.array(bdf),
        x=bdf.precision,
        y=bdf.recall,
        custom_data=[bdf.index, bdf['color']],
        color=bdf.color,
        color_discrete_sequence=px.colors.qualitative.D3
    )

fig.update_layout(
    title=f"Predicting {test}",
    xaxis_title="Precision",
    yaxis_title="Recall",
    xaxis_range=[-0.05, 1.05],
    yaxis_range=[-0.05, 1.05],
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)


im1 = bdf.index[0]
fig2 = px.scatter(np.array(bdf.loc[im1]),
        x=bdf.loc[im1]['precision'],
        y=bdf.loc[im1]['recall'],
        custom_data=[bdf.loc[im1].index, bdf.loc[im1]['color']],
        color=bdf.loc[im1]['color'],
        color_discrete_sequence=px.colors.qualitative.D3
    )

fig2.update_layout(
    title=f"Predicting {im1}",
    xaxis_title="Precision",
    yaxis_title="Recall",
    legend_title="Legend Title",
    xaxis_range=[-0.05, 1.05],
    yaxis_range=[-0.05, 1.05],
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
fig2.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)


app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True, style={"width": "80%", "height": "800px"}),
        dcc.Tooltip(id="graph-tooltip"),
    ]),
    html.Div([
        dcc.Graph(id="graph-basic-3", figure=fig2, clear_on_unhover=True, style={"width": "80%", "height": "800px"}),
        dcc.Tooltip(id="graph-detail"),
    ]),
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData")
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    bbox = hoverData['points'][0]["bbox"]
    img_src = hoverData['points'][0]['customdata'][0]
    model = hoverData['points'][0]['customdata'][1]
    pred_path = rf"D:\data\solardnn\{model}\predictions\{model}_resnet34_42_v1_predicting_{test}\pred_masks"
    img_dat = np_image_to_base64(img_path,mask_path, pred_path, img_src)
    children = [
        html.Div([html.H1(f"{model}"),
                  html.H6(f"{hoverData['points'][0]['x']}"),
                  html.H6(f"{hoverData['points'][0]['y']}"),
                  html.H6(f"{img_src}")
                  ]),
        html.Div([
            html.Img(src=img_dat, style={"width": "100%"}),
        ], style={'width': '400px', 'white-space': 'normal'})
    ]

    return True, bbox, children

@app.callback(
    Output("graph-detail", "show"),
    Output("graph-detail", "bbox"),
    Output("graph-detail", "children"),
    Input("graph-basic-3", "hoverData")
)
def display_hover2(hoverData):
    return display_hover(hoverData)


@app.callback(
    Output('graph-basic-3', 'figure'),
    Input('graph-basic-2', 'clickData')
)
def update_figure(selected_pt):
    if selected_pt is None:
        return no_update

    img_src = selected_pt['points'][0]['customdata'][0]
    df_filt = bdf.loc[img_src]

    thisfig = px.scatter(np.array(df_filt),
        x=df_filt['precision'],
        y=df_filt['recall'],
        custom_data=[df_filt.index, df_filt['color']],
        color=df_filt['color'],
        color_discrete_sequence=px.colors.qualitative.D3
    )

    thisfig.update_layout(
        title=f"Predicting {img_src}",
        xaxis_title="Precision",
        yaxis_title="Recall",
        xaxis_range=[-0.05, 1.05],
        yaxis_range=[-0.05, 1.05],
        legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    thisfig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    return thisfig


if __name__ == "__main__":
    app.run_server(debug=True)