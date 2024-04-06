import dash
import dash.dependencies as dd
from dash import dcc as dcc
import dash_bootstrap_components as dbc
from dash import html
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import preprocessing_text as pt
# from underthesea import word_tokenize, pos_tag, sent_tokenize
# import regex

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

from io import BytesIO

import pandas as pd
from wordcloud import WordCloud
import base64

from joblib import dump
from joblib import load

#css 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# load model
model = load('models\svm_project1.joblib')

# load data
df = pd.read_csv('Data/2_Reviews.csv')

def plot_wordcloud(data):
    d = {a: x for a, x in data.values}
    wc = WordCloud(background_color='black', width=480, height=360)
    wc.fit_words(d)
    return wc.to_image()

STYLE_={
    # 'border': '1px solid #000',  # Đặt viền đen độ dày 1px
    'padding': '20px',  # Đặt lề bên trong là 10px
    'border-radius': '10px',  # Đặt viền cong với bán kính 5px,
    'display': 'inline-block',
    # 'color': 'white',
    # 'background-color': 'teal',
    'width': '97%',
    'height':'100%',
    'text-align': 'center',
}

app.layout = html.Div([
    html.H2('Nhập id nhà hàng:', style={'textAlign': 'center'}),
    dcc.Input(id='id', type='text', value='', style={'textAlign': 'center'}),
    html.H2('Nhập comment:', style={'textAlign': 'center'}),
    dcc.Input(id='comment', type='text', value='', style={'textAlign': 'center', 'width': '50%', 'height': '100px'}),
    html.Br(),
    html.Br(),
    html.Button('Submit', id='submit', style={'textAlign': 'center', 'background-color': 'teal', 'color': 'white', 'padding_top': '30px'}),
    html.Br(),
    html.Br(),
    html.Div(id='output', style={'fontSize': 20}),
    dcc.Graph(id='histogram_rate'),
    dcc.Graph(id='pie-chart'),
    dbc.Row([
        dbc.Col([
            html.H2('', id = 'wc-neg-title',style={'textAlign': 'center'}),
            html.Div([dcc.Graph(id='wordcloud-graph-neg')])
        ], style={'margin': '20px'}),
        dbc.Col([
            html.H2('', id = 'wc-pos-title', style={'textAlign': 'center'}),
            html.Div([dcc.Graph(id='wordcloud-graph-pos')])
        ], style={'margin': '20px'}),
    #     dbc.Col([
    #         # html.H2('Wordcloud', style={'textAlign': 'center'}),
    #         html.Div([dcc.Graph(id='wordcloud-graph-neu')])
    #     ], style={'margin': '20px'})
    ], style={'display': 'flex', 'justifyContent': 'center'}),
], style=STYLE_)

# hàm xử lý sự kiện click nút submit
@app.callback(
    dd.Output('output', 'children'),
    [dd.Input('submit', 'n_clicks')],
    [dd.State('id', 'value'),
     dd.State('comment', 'value')]
)

# print id và comment
def update_output(n_clicks, id, comment):
    if n_clicks is None:
        return ''
    print(f'id: {id}, comment: {comment}')
    return 'Đánh giá của bạn là: ' + str(model.predict([comment])[0])

@app.callback(
    [dd.Output('histogram_rate', 'figure'),
     dd.Output('pie-chart', 'figure'),
    dd.Output('wordcloud-graph-neg', 'figure'),
    dd.Output('wordcloud-graph-pos', 'figure'),
    # dd.Output('wordcloud-graph-neu', 'figure')
    dd.Output('wc-neg-title', 'children'),
    dd.Output('wc-pos-title', 'children'),
     ],
    [dd.Input('submit', 'n_clicks')],
    [dd.State('id', 'value')]
)

def update_pie_chart(n_clicks, id):
    if id == '' or n_clicks is None:
        fig = go.Figure()
        fig.update_layout(title='', paper_bgcolor='white', plot_bgcolor='white')
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        return [fig, fig, fig, fig, '', '']
    
    # query data by IDRestaurant
    data = df[df['IDRestaurant'] == int(id)]

    # predict review
    pred = model.predict(data['Comment'])
    res = np.unique(pred, return_counts=True)
    # add column pred
    data['pred'] = pred

    # histogram char
    x_values = data['Rating'].value_counts().index
    y_values = data['Rating'].value_counts().values
    fig_hist = go.Bar(x = x_values, y = y_values)
    layout_hist = go.Layout(title={'text':'Tổng quan các lượt ratings và phản hồi về nhà hàng', 'font': {'size': 30}}, xaxis={'title': 'Rating'}, yaxis={'title': 'Số lượt đánh giá'})

    # pie chart
    fig_pie = [go.Pie(labels=res[0], values=res[1])]
    layout_pie = go.Layout(title={'text':'Tỉ lệ các loại đánh giá', 'font': {'size': 30}})

    # wordcloud
    # text = ' '.join(data['Comment'])
    text_neg = ' '.join(data[data['pred'] == 'Negative']['Comment'])
    text_pos = ' '.join(data[data['pred'] == 'Positive']['Comment'])

    # clean text loại bỏ stopword
    text_neg = pt.optimized_process_text(text_neg, pt.stopwords_lst)
    text_pos = pt.optimized_process_text(text_pos, pt.stopwords_lst)

    # text_neu = ' '.join(data[data['pred'] == 'Neutral']['Comment'])


    wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(text_neg)
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(text_pos)
    # wordcloud_neu = WordCloud(width=800, height=400, background_color='white').generate(text_neu)
    # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    # wordcloud.to_file("wordcloud.png")
    wordcloud_neg.to_file("wordcloud_neg.png")
    wordcloud_pos.to_file("wordcloud_pos.png")
    # wordcloud_neu.to_file("wordcloud_neu.png")

    # Đọc tệp ảnh và mã hóa dưới dạng base64
    # with open("wordcloud.png", "rb") as img_file:
    #     encoded_img = base64.b64encode(img_file.read()).decode()

    with open("wordcloud_neg.png", "rb") as img_file:
        encoded_img_neg = base64.b64encode(img_file.read()).decode()

    with open("wordcloud_pos.png", "rb") as img_file:
        encoded_img_pos = base64.b64encode(img_file.read()).decode()

    # with open("wordcloud_neu.png", "rb") as img_file:
    #     encoded_img_neu = base64.b64encode(img_file.read()).decode()

    # Tạo một đối tượng go.Figure chứa hình ảnh WordCloud
    fig_wc_neg = go.Figure()
    fig_wc_neg.add_layout_image(
        source='data:image/png;base64,{}'.format(encoded_img_neg),
        xref="paper", yref="paper",
        x=0, y=1,
        sizex=1, sizey=1,
        sizing="stretch",
        opacity=1,
        layer="below"
    )
    fig_wc_neg.update_xaxes(visible=False)
    fig_wc_neg.update_yaxes(visible=False)
    fig_wc_neg.update_layout(
        width=800,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig_wc_pos = go.Figure()
    fig_wc_pos.add_layout_image(
        source='data:image/png;base64,{}'.format(encoded_img_pos),
        xref="paper", yref="paper",
        x=0, y=1,
        sizex=1, sizey=1,
        sizing="stretch",
        opacity=1,
        layer="below"
    )
    fig_wc_pos.update_xaxes(visible=False)
    fig_wc_pos.update_yaxes(visible=False)
    fig_wc_pos.update_layout(
        width=800,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # fig_wc_neu = go.Figure()
    # fig_wc_neu.add_layout_image(
    #     source='data:image/png;base64,{}'.format(encoded_img_neu),
    #     xref="paper", yref="paper",
    #     x=0, y=1,
    #     sizex=1, sizey=1,
    #     sizing="stretch",
    #     opacity=1,
    #     layer="below"
    # )
    # fig_wc_neu.update_xaxes(visible=False)
    # fig_wc_neu.update_yaxes(visible=False)
    # fig_wc_neu.update_layout(
    #     width=800,
    #     height=400,
    #     margin=dict(l=0, r=0, t=0, b=0)
    # )

    return [go.Figure(data=fig_hist, layout=layout_hist), 
            go.Figure(data=fig_pie, layout=layout_pie), 
            # 'data:image/png;base64,{}'.format(encoded_img)]
            go.Figure(data=fig_wc_neg, layout=layout_pie),
            go.Figure(data=fig_wc_pos, layout=layout_pie),
            'WordCloud của các lượt review Negative',
            'WordCloud của các lượt review Positive',
            # go.Figure(data=fig_wc_neu, layout=layout_pie)
            ]

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='localhost')