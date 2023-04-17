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

test = "NY-Q"

data_path = rf"D:\data\solardnn\results\resnet34_42_1\{test}_test\resnet34_42_v1_{test}_imgmetrics.xlsx"
img_path = rf"D:\data\solardnn\{test}\tiles\img"
mask_path = rf"D:\data\solardnn\{test}\tiles\mask"



color = {"CA-S": 0, "CA-F": 1, "FR-G": 2, "FR-I": 3, "DE-G": 4, "NY-Q": 5, "CMB-6": 6}

def np_image_to_base64(im_path, msk_path, pre_path, im_name, alph=0.75):
    im = Image.open(os.path.join(im_path, im_name))

    red = Image.new('RGB', im.size, (255, 0, 0))
    ms = Image.open(os.path.join(msk_path, im_name)).convert('L').point(lambda i: i*alph)
    im = Image.composite(red, im, ms).convert('RGB')

    blue = Image.new('RGB', im.size, (0, 0, 255))
    pr = Image.open(os.path.join(pre_path, im_name)).convert('L').point(lambda i: i*alph).resize(im.size)
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

bdf = pd.DataFrame({"precision": ps, "recall": rs, "color": cs}, index=ims)

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
fig = px.scatter(
        x=bdf['precision'],
        y=bdf['recall'],
        color=bdf['color'],
        color_discrete_sequence=px.colors.qualitative.D3
    )

fig.update_layout(
    title=f"Predicting {test}",
    xaxis_title="Precision",
    yaxis_title="Recall",
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


app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True, style={"width": "80%", "height": "800px"}),
    dcc.Tooltip(id="graph-tooltip"),
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]+pt['curveNumber']*200

    df_row = bdf.iloc[num]
    img_src = df_row.name
    mod = df_row['color']
    pred_path = rf"D:\data\solardnn\{mod}\predictions\{mod}_resnet34_42_v1_predicting_{test}\pred_masks"
    img_dat = np_image_to_base64(img_path,mask_path, pred_path, img_src)
    children = [
        html.Div([html.H1(f"{mod}"),
                  html.H6(f"{num}"),
                  html.H6(f"{df_row['precision']}"),
                  html.H6(f"{df_row['recall']}"),
                  html.H6(f"{img_src}")
                  ]),
        html.Div([
            html.Img(src=img_dat, style={"width": "100%"}),
        ], style={'width': '400px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=True)