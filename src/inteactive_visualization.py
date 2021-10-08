# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:03:13 2021

@author: Panagiotis Anagnostou
"""

import copy
import dash
import json
import matplotlib
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import socket
import webbrowser

from contextlib import closing
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from skimage import io
from tempfile import NamedTemporaryFile

external_stylesheets =  ['./assets/int_viz.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def main(inputData):
    global app, X
    
    app_layout(app=app, input_object=inputData)
    
    app.run_server(debug=False) # , dev_tools_ui=False, dev_tools_props_check=False, suppress_callback_exceptions=True)

#%% Callbacks

@app.callback(Output('figure_panel', 'children'),
              Input('url', 'pathname'),
              State('cache_dump', 'data'))
def display(pathname, data):
    '''URL manipulation callback.'''
    data = json.loads(data)
    
    if pathname == '/clustering_results':
        div = Cluster_Scatter_Plot(data['new_input_object'])
    elif pathname == '/delete_split':
        div = Delete_Split_Plot(data['new_input_object'], data['figure_path'])
    elif pathname == '/splitpoint_manipulation':
        div = Splitpoint_Manipulation(data['new_input_object'])
    else:
        div = Cluster_Scatter_Plot(data['input_object'])
        
    return div


@app.callback(Output('splitpoint_Main', 'children'),
              Output('splitView', 'max'),
              Output('splitView', 'marks'),
              Output('splitpoint_Manipulation', 'value'),
              Output('splitpoint_Manipulation', 'min'),
              Output('splitpoint_Manipulation', 'max'),
              State('cache_dump', 'data'),
              State('splitpoint_Scatter', 'figure'),
              Input('splitView', 'value'),
              State('splitView', 'max'),
              State('splitView', 'marks'),
              Input('splitpoint_Manipulation', 'value'),
              State('splitpoint_Manipulation', 'min'),
              State('splitpoint_Manipulation', 'max'),
              Input('splitpoint_Man_apply', 'n_clicks'), prevent_initial_call=True)
def Splitpoint_Manipulation_Callback(data, curent_figure, split_number, maximum_number_splits, split_marks, splitpoint_position, splitpoint_minimum, splitpoint_max, apply_button):
    callback_context = dash.callback_context
    callback_ID = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    data = json.loads(data)
    
    if callback_ID == 'splitView':
        data_matrix, splitpoint, internal_nodes, number_of_nodes = data_preparation(data['new_input_object'], split_number)
        
        splitMax = maximum_number_splits
        splitMarks = split_marks
        
        splitPMin = data_matrix['PC1'].min()
        splitPMax = data_matrix['PC1'].max()
        
        category_order = {'cluster': [ str(i) for i in range(len(np.unique(data_matrix['cluster']))) ]}
        color_map = matplotlib.cm.get_cmap('tab20', number_of_nodes)
        colList = { str(i): convert_to_hex(color_map(i)) for i in range(color_map.N) }
        
        curent_figure = px.scatter(data_matrix, x="PC1", y="PC2", color="cluster", hover_name="cluster", category_orders=category_order, color_discrete_map=colList)
        curent_figure.add_shape(type='line', yref="y", xref="x", xsizemode= 'scaled', ysizemode= 'scaled', x0=splitpoint, y0=data_matrix['PC2'].min()*1.2, x1=splitpoint, y1=data_matrix['PC2'].max()*1.2, line=dict(color='red', width=1.5))
        
        curent_figure.update_layout(width=850, height=650, plot_bgcolor='#fff')
        curent_figure.update_traces(mode="markers",  marker=dict(size=4), hovertemplate=None, hoverinfo='skip')
        curent_figure.update_xaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
        curent_figure.update_yaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
        
    elif callback_ID == 'splitpoint_Manipulation':
        curent_figure = go.Figure(data = curent_figure['data'], layout = go.Layout(curent_figure['layout']))
        
        curent_figure.update_shapes({'x0': splitpoint_position, 'x1': splitpoint_position})
        
        splitMax = maximum_number_splits
        splitMarks = split_marks
        
        splitpoint = splitpoint_position
        splitPMin = splitpoint_minimum
        splitPMax = splitpoint_max
        
    elif callback_ID == 'splitpoint_Man_apply':
        with open(data['new_input_object'], 'rb') as obj_file:
            obj = pickle.load(obj_file)
        
        obj = obj.recalculate_after_spchange(split_number, splitpoint_position)
        
        with open(data['new_input_object'], 'wb') as obj_file: 
            pickle.dump(obj, obj_file)
        
        data_matrix, splitpoint, internal_nodes, number_of_nodes = data_preparation(data['new_input_object'], split_number)
        
        splitMax = len(internal_nodes) - 1
        splitMarks = {str(i): str(i) for i in range(len(internal_nodes))}
        
        splitPMin = data_matrix['PC1'].min()
        splitPMax = data_matrix['PC1'].max()
        
        category_order = {'cluster': [ str(i) for i in range(len(np.unique(data_matrix['cluster']))) ]}
        color_map = matplotlib.cm.get_cmap('tab20', number_of_nodes)
        colList = { str(i): convert_to_hex(color_map(i)) for i in range(color_map.N) }
        
        curent_figure = px.scatter(data_matrix, x="PC1", y="PC2", color="cluster", hover_name="cluster", category_orders=category_order, color_discrete_map=colList)
        curent_figure.add_shape(type='line', yref="y", xref="x", xsizemode= 'scaled', ysizemode= 'scaled', x0=splitpoint, y0=data_matrix['PC2'].min()*1.2, x1=splitpoint, y1=data_matrix['PC2'].max()*1.2, line=dict(color='red', width=1.5))
        
        curent_figure.update_layout(width=850, height=650, plot_bgcolor='#fff')
        curent_figure.update_traces(mode="markers",  marker=dict(size=4), hovertemplate=None, hoverinfo='skip')
        curent_figure.update_xaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
        curent_figure.update_yaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
        
    else:
        curent_figure = go.Figure(data = curent_figure['data'], layout = go.Layout(curent_figure['layout']))
        
        splitMax = maximum_number_splits
        splitMarks = split_marks
        
        splitpoint = splitpoint_position
        splitPMin = splitpoint_minimum
        splitPMax = splitpoint_max
        
    return [dcc.Graph(id='splitpoint_Scatter', figure=curent_figure, config={'displayModeBar': False})], splitMax, splitMarks, splitpoint, splitPMin, splitPMax


@app.callback(Output('delete_split_main', 'children'),
              Output('delete_split_number', 'options'),
              State('cache_dump', 'data'),
              State('delete_split_number', 'value'),
              Input('delete_split_apply', 'n_clicks'), prevent_initial_call=True)
def DSCallback(data, value, button):
    data = json.loads(data)
    
    with open(data['new_input_object'], 'rb') as obj_file:
        obj = pickle.load(obj_file)
        
    obj.delete_split(value)
    
    with open(data['new_input_object'], 'wb') as obj_file: 
        pickle.dump(obj, obj_file)
    
    _, _, internal_nodes, _ = data_preparation(data['new_input_object'], 0)
    
    obj.tree_data_viz(name=os.path.basename(data['figure_path']).split('.')[0], directory=os.path.dirname(data['figure_path']), format="png", color_map="tab20", nodeLabels=True, splitPointVisible=False)
    
    image = io.imread(data['figure_path'])
    
    ratio = image.shape[0] / image.shape[1]
    width = 900 if image.shape[1] > 500 else 600
    new_figure = px.imshow(image, width=width, height=width * (ratio - 0.05))
    new_figure.update_layout(plot_bgcolor='#fff')
    new_figure.update_traces(hovertemplate=None, hoverinfo='skip')
    new_figure.update_xaxes(showticklabels=False, fixedrange=True).update_yaxes(showticklabels=False, fixedrange=True) 
    
    delete_split_node_options = [{'label': 'Split ' + str(i), 'value': i} for i in range(len(internal_nodes))]
    
    return [dcc.Graph(figure=new_figure, config={'displayModeBar': False})], delete_split_node_options


#%% main layout of the app
def app_layout(app, input_object): 
    tmpFilesWrapers = {"input_object": NamedTemporaryFile("wb+", suffix='.inv', delete=False),
                       "new_input_object": NamedTemporaryFile("wb+", suffix='.inv', delete=False),
                       "figure_path": NamedTemporaryFile("wb+", suffix='.png', delete=False)}
    tmpFileNames = {"input_object": tmpFilesWrapers['input_object'].name,
                    "new_input_object": tmpFilesWrapers['new_input_object'].name,
                    "figure_path": tmpFilesWrapers['figure_path'].name}
    
    with open(tmpFileNames['input_object'], 'wb') as input_object_file: 
        pickle.dump(input_object, input_object_file)
        
    with open(tmpFileNames['new_input_object'], 'wb') as new_input_object_file: 
        pickle.dump(input_object, new_input_object_file)
        
    app.layout = html.Div(children=[
        dcc.Location(id='url', refresh=False),
        
        html.H1("dePDDP: Interactive Visualisation"),
        
        html.Ul(children=[
            html.Li(dcc.Link('Clustgering results', href='/clustering_results'), style={'display': 'inline', 'margin': '0px 5px'}),
            html.Li(dcc.Link('Split Point Manipulation', href='/splitpoint_manipulation'), style={'display': 'inline', 'margin': '0px 5px'}),
            html.Li(dcc.Link('Delete Split', href='/delete_split'), style={'display': 'inline', 'margin': '0px 5px'})],
            style={'margin': '60px 0px 30px 0px'}),
        html.Br(),
        # html.Div(id='debug',  style={'width': '685px', 'padding': '20px 150px 20px 80px'}),
        
        html.Div(id='figure_panel'),
        
        dcc.Store(id='cache_dump', data=str(json.dumps(tmpFileNames)))
    ], style={'min-height': '100%', 'width': '900px', 'text-align': 'center', 'margin': 'auto'})


#%% home page
def Cluster_Scatter_Plot(object_path):
    data_matrix, _, _, number_of_nodes = data_preparation(object_path, 0)
    
    category_order = {'cluster': [ str(i) for i in range(len(np.unique(data_matrix['cluster']))) ]}
    color_map = matplotlib.cm.get_cmap('tab20', number_of_nodes)
    colList = { str(i): convert_to_hex(color_map(i)) for i in range(color_map.N) }
    
    figure = px.scatter(data_matrix, x="PC1", y="PC2", color="cluster", category_orders=category_order, color_discrete_map=colList)
    figure.update_layout(width=850, height=650, plot_bgcolor='#fff')
    figure.update_traces(mode="markers",  marker=dict(size=4), hovertemplate=None, hoverinfo='skip')
    figure.update_xaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
    figure.update_yaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
    
    description = dcc.Markdown('''
            PCA visualization of the data with coloring on the extracted clusters.
            ''')
    
    return html.Div([description, dcc.Graph(figure=figure, config={'displayModeBar': False})])


#%% Split-point manipulation
def Splitpoint_Manipulation(object_path):
    data_matrix, splitpoint, internal_nodes, number_of_nodes = data_preparation(object_path, 0)
    
    category_order = {'cluster': [ str(i) for i in range(len(np.unique(data_matrix['cluster']))) ]}
    color_map = matplotlib.cm.get_cmap('tab20', number_of_nodes)
    colList = { str(i): convert_to_hex(color_map(i)) for i in range(color_map.N) }
    
    figure = px.scatter(data_matrix, x="PC1", y="PC2", color="cluster", hover_name="cluster", category_orders=category_order, color_discrete_map=colList)
    figure.add_shape(type='line', yref="y", xref="x", xsizemode= 'scaled', ysizemode= 'scaled', x0=splitpoint, y0=data_matrix['PC2'].min()*1.2, x1=splitpoint, y1=data_matrix['PC2'].max()*1.2, line=dict(color='red', width=1.5))
    
    figure.update_layout(width=850, height=650, plot_bgcolor='#fff')
    figure.update_traces(mode="markers",  marker=dict(size=4), hovertemplate=None, hoverinfo='skip')
    figure.update_xaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
    figure.update_yaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
    
    description = dcc.Markdown('''
            Select the split and the splitpoint in order to manipulate one of the dePDDP`s substeps.
            ''')
    
    splitStep = html.Div(['Select split: ',dcc.Slider(id='splitView', min=0, max=len(internal_nodes)-1, value=0, marks={str(i): str(i) for i in range(len(internal_nodes))}, step=None)],
                         style={'text-align': 'left', 'width': '740px', 'padding': '20px 80px 20px 80px'})
    
    scatter = html.Div(id='splitpoint_Main', children=[dcc.Graph(id='splitpoint_Scatter', figure=figure, config={'displayModeBar': False})])
    
    spSlider = html.Div(['Select splitpoint: ',dcc.Slider(id='splitpoint_Manipulation', min=data_matrix['PC1'].min() - 0.001, max=data_matrix['PC1'].max() + 0.001, step=0.0001, value=splitpoint)], style={'text-align': 'left', 'width': '659px', 'margin': '0px', 'padding': '0px 0px 0px 92px'})
    
    applyButton = html.Button('Apply', id='splitpoint_Man_apply', n_clicks=0)
    
    return html.Div([description, splitStep, scatter, spSlider, applyButton])


#%% Delete Split
def Delete_Split_Plot(object_path, figure_path):
    
    _, _, internal_nodes, _ = data_preparation(object_path, 0)
    
    with open(object_path, 'rb') as obj_file:
        obj = pickle.load(obj_file)
    
    obj.tree_data_viz(name=os.path.basename(figure_path).split('.')[0], directory=os.path.dirname(figure_path), format="png", color_map="tab20", nodeLabels=True, splitPointVisible=False)
    
    img = io.imread(figure_path)
    
    ratio = img.shape[0] / img.shape[1]
    width = 900 if img.shape[1] > 500 else 450
    figure = px.imshow(img, width=width, height=width * (ratio - 0.05))
    figure.update_layout(plot_bgcolor='#fff')
    figure.update_traces(hovertemplate=None, hoverinfo='skip')
    figure.update_xaxes(showticklabels=False, fixedrange=True).update_yaxes(showticklabels=False, fixedrange=True) 
    
    description = dcc.Markdown('''
            Select the split and delete it.
            ''')
            
    delSplDropdown = html.Div(dcc.Dropdown(id='delete_split_number', options=[{'label': 'Split ' + str(i), 'value': i} for i in range(len(internal_nodes))]), style={'width': '300px', 'height': '32px'})
    applyButton = html.Div(html.Button('Apply', id='delete_split_apply', n_clicks=0), style={'height': '32px'})
    table = html.Table([ html.Tr(html.Td(['Select split: '], style={'text-align': 'left'})),
        html.Tr([html.Td(delSplDropdown), html.Td(applyButton)]) ])
    
    nodesSplit = html.Div(id='delete_split_main', children=[dcc.Graph(figure=figure, config={'displayModeBar': False})])
    
    return html.Div([description,table, nodesSplit])


#%% Util Functions
def data_preparation(object_path, splitVal):
    with open(object_path, 'rb') as obj_file:
        tree = pickle.load(obj_file).tree
        
    clusters = tree.leaves()
    clusters = sorted(clusters, key = lambda x:x.identifier)
    
    root = tree.get_node(0)
    
    cluster_map = np.zeros(len(root.data['indices']))
    for i in clusters:
        cluster_map[i.data['indices']] = str(int(i.identifier))
    
    dictionary_of_nodes = tree.nodes
    number_of_nodes = len(list(dictionary_of_nodes.keys()))
    leaf_node_list = [ j.identifier for j in clusters ]
    internal_nodes = [ i for i in range(number_of_nodes )if not (i in leaf_node_list) ]
    
    if len(internal_nodes) == 0: internal_nodes = [0]
    
    node_to_visualize = tree.get_node(internal_nodes[splitVal]) if len(internal_nodes) != 0 else tree.get_node(0)
    
    data_matrix = pd.DataFrame({
        'PC1': node_to_visualize.data['projection'][:,0],
        'PC2': node_to_visualize.data['projection'][:,1],
        'cluster': cluster_map[node_to_visualize.data['indices']]
        })
    
    data_matrix['cluster'] = data_matrix['cluster'].astype(int).astype(str)
    splitpoint = node_to_visualize.data['splitpoint']
    
    return data_matrix, splitpoint, internal_nodes, number_of_nodes


def convert_to_hex(rgba_color) :
    red = int(rgba_color[0]*255)
    green = int(rgba_color[1]*255)
    blue = int(rgba_color[2]*255)
    return '#{r:02x}{g:02x}{b:02x}'.format(r=red,g=green,b=blue)
    
 





 
 
 
 
 
 