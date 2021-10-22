# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:03:13 2021

@author: Panagiotis Anagnostou
"""

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
from flask import request
from skimage import io
from tempfile import NamedTemporaryFile
from treelib import Tree

import src.visualizations as viz


external_stylesheets =  ['./assets/int_viz.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def main(inputData):
    """
    Main function of the interactive visualization. 
    
    Parameters
    ----------
    inputData : dePDDPcl.dePDDP object
        The dePDDP object that deems visualization.
    
    app_layout(app=app, input_object=inputData)
    Returns
    -------
    manipullated : dePDDPcl.dePDDP object
        The manipullated by the visulization object.
    
    """
    
    # Create temp files for smooth execution of the inteactive visualization
    tmpFilesWrapers = {"input_object": NamedTemporaryFile("wb+", suffix='.inv', delete=False),
                       "new_input_object": NamedTemporaryFile("wb+", suffix='.inv', delete=False),
                       "figure_path": NamedTemporaryFile("wb+", suffix='.png', delete=False)}
    tmpFileNames = {"input_object": tmpFilesWrapers['input_object'].name,
                    "new_input_object": tmpFilesWrapers['new_input_object'].name,
                    "figure_path": tmpFilesWrapers['figure_path'].name}
    
    # load the necessary data on the temp files
    with open(tmpFileNames['input_object'], 'wb') as input_object_file: 
        pickle.dump(inputData, input_object_file)
        
    with open(tmpFileNames['new_input_object'], 'wb') as new_input_object_file: 
        pickle.dump(inputData, new_input_object_file)
    
    # Create and run the server app
    app_layout(app=app, tmpFileNames=tmpFileNames)
    webbrowser.open('http://localhost:8050', new=2)
    app.run_server(debug=False)
    
    # Recall the object to return before temp file deletion
    with open(tmpFileNames['new_input_object'], 'rb') as obj_file:
        obj = pickle.load(obj_file)
    
    # Close and delete all temp files
    tmpFilesWrapers["input_object"].close()
    tmpFilesWrapers["new_input_object"].close()
    tmpFilesWrapers["figure_path"].close()
    
    os.unlink(tmpFileNames["input_object"])
    os.unlink(tmpFileNames["new_input_object"])
    os.unlink(tmpFileNames["figure_path"])
    
    assert not os.path.exists(tmpFileNames["input_object"])
    assert not os.path.exists(tmpFileNames["new_input_object"])
    assert not os.path.exists(tmpFileNames["figure_path"])
    
    return obj

#%% Callbacks
@app.callback(Output('figure_panel', 'children'),
              Input('url', 'pathname'),
              State('cache_dump', 'data'))
def display_menu(pathname, data):
    '''
    Menu interchange callback.
    
    Parameters
    ----------
    pathname : str
        The new url pathname that the webpage needs to go to.
    
    data : dict
        The paths of the temporary files created for the execution of the interactive visualization.
    
    Returns
    -------
    div : dash.html
        A div containing all the visual components needed for each of the pages menu.
    
    '''
    data = json.loads(data)
    
    if pathname == '/clustering_results':
        div = Cluster_Scatter_Plot(data['new_input_object'])
    # elif pathname == '/delete_split':
    #     div = Delete_Split_Plot(data['new_input_object'], data['figure_path'])
    elif pathname == '/splitpoint_manipulation':
        div = Splitpoint_Manipulation(data['new_input_object'])
    elif pathname == '/shutdown':
        div = html.Div(children=[
            html.Br(),
            html.H4("Close the Window!!!")])
        shutdown()
    else:
        div = Cluster_Scatter_Plot(data['input_object'])
        
        with open(data['input_object'], 'rb') as obj_file:
            obj = pickle.load(obj_file)
        
        with open(data['new_input_object'], 'wb') as obj_file: 
            pickle.dump(obj, obj_file)
        
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
    """
    Funciton trigered by the change in valua on the Input elements. The trigers are:
        
        1. the change in split of the data.
        2. the changes in the split point.
        3. the apply button for the change to the new split point (partial algorithm execution from the sellecte split).
        
    The rest of the function's inputs are nputs that can't triger this callback but their data are necessary for this callbacks execution.
    
    Parameters
    ----------
        data : dict
            The paths of the temporary files created for the execution of the interactive visualization.
        
        curent_figure : dict 
            A dictionary created from the plotly express object plots. It is used for the manipullation of the current figure.
        
        split_number : int
            The value of the split to project (or is projected, depending on the callback context trigered state).
        
        maximum_number_splits : int
            The number of splits there are in the mdoel created / manipullated.
            
        split_marks : dict
            The split number as key of the dictionary and the assinged value as value of the dictionary.
            
        splitpoint_position : float
            The current position of the shape that repressentes the splitpoint of the currently sellected split, extracted from the splitpoint slider.
        
        splitpoint_minimum : float
            The minimum value that can be assigned to the splitpoint, extracted from the splitpoint slider.
        
        splitpoint_max : float
            The maximum value that can be assigned to the splitpoint, extracted from the splitpoint slider.
        
        apply_button : int
            The number of clicks of the apply button (Not needed in the function`s execution, but necessary for the callback definition).
    
    Returns
    -------
        figure : dash.dcc.Graph object
            A figure that can be intergraded at dash`s html componentes.
        
        splitMax : float
            The new value for the maximum split number.
        
        splitMarks : float
            The changed marks the spit can utilize as values.
       
        splitpoint : float
            The new splitpoint generated by the callback.
        
        splitPMin : int
            The minimum value the splitpoint can take based on the callback.
        
        splitPMax : int
            The maximum value the splitpoint can take based on the callback.
    """
    callback_context = dash.callback_context
    callback_ID = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    # load cache dumped files
    data = json.loads(data)
    
    # Basic check on the callback triger, based on the dash html elements ID.
    if callback_ID == 'splitView':
        data_matrix, splitpoint, internal_nodes, number_of_nodes = data_preparation(data['new_input_object'], split_number)
        
        # ensure correct values for the sliders
        splitMax = maximum_number_splits
        splitMarks = split_marks
        
        splitPMin = data_matrix['PC1'].min()
        splitPMax = data_matrix['PC1'].max()
        
        # create visualization points
        category_order = {'cluster': [ str(i) for i in range(len(np.unique(data_matrix['cluster']))) ]}
        color_map = matplotlib.cm.get_cmap('tab20', number_of_nodes)
        colList = { str(i): convert_to_hex(color_map(i)) for i in range(color_map.N) }
        
        # create scatter plot with the splitpoint shape
        curent_figure = px.scatter(data_matrix, x="PC1", y="PC2", color="cluster", hover_name="cluster", category_orders=category_order, color_discrete_map=colList)
        curent_figure.add_shape(type='line', yref="y", xref="x", xsizemode= 'scaled', ysizemode= 'scaled', x0=splitpoint, y0=data_matrix['PC2'].min()*1.2, x1=splitpoint, y1=data_matrix['PC2'].max()*1.2, line=dict(color='red', width=1.5))
        
        # reform visualization
        curent_figure.update_layout(width=850, height=650, plot_bgcolor='#fff')
        curent_figure.update_traces(mode="markers",  marker=dict(size=4), hovertemplate=None, hoverinfo='skip')
        curent_figure.update_xaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
        curent_figure.update_yaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
        
    elif callback_ID == 'splitpoint_Manipulation':
        # reconstruct the already created figure as figure from dict
        curent_figure = go.Figure(data = curent_figure['data'], layout = go.Layout(curent_figure['layout']))
        
        # update the splitpoint shape location
        curent_figure.update_shapes({'x0': splitpoint_position, 'x1': splitpoint_position})
        
        # ensure correct values for the sliders
        splitMax = maximum_number_splits
        splitMarks = split_marks
        
        splitpoint = splitpoint_position
        splitPMin = splitpoint_minimum
        splitPMax = splitpoint_max
        
    elif callback_ID == 'splitpoint_Man_apply':
        # execution of the splitpoint manipulation algorithmically
        with open(data['new_input_object'], 'rb') as obj_file:
            obj = pickle.load(obj_file)
        
        obj = recalculate_after_spchange(obj, split_number, splitpoint_position)
        
        with open(data['new_input_object'], 'wb') as obj_file: 
            pickle.dump(obj, obj_file)
        
        # reconstrauction of the figure and its slider from scrach
        data_matrix, splitpoint, internal_nodes, number_of_nodes = data_preparation(data['new_input_object'], split_number)
        
        # ensure correct values for the sliders
        splitMax = len(internal_nodes) - 1
        splitMarks = {str(i): str(i) for i in range(len(internal_nodes))}
        
        splitPMin = data_matrix['PC1'].min()
        splitPMax = data_matrix['PC1'].max()
        
        # create visualization points
        category_order = {'cluster': [ str(i) for i in range(len(np.unique(data_matrix['cluster']))) ]}
        color_map = matplotlib.cm.get_cmap('tab20', number_of_nodes)
        colList = { str(i): convert_to_hex(color_map(i)) for i in range(color_map.N) }
        
        # create scatter plot with the splitpoint shape
        curent_figure = px.scatter(data_matrix, x="PC1", y="PC2", color="cluster", hover_name="cluster", category_orders=category_order, color_discrete_map=colList)
        curent_figure.add_shape(type='line', yref="y", xref="x", xsizemode= 'scaled', ysizemode= 'scaled', x0=splitpoint, y0=data_matrix['PC2'].min()*1.2, x1=splitpoint, y1=data_matrix['PC2'].max()*1.2, line=dict(color='red', width=1.5))
        
        # reform visualization
        curent_figure.update_layout(width=850, height=650, plot_bgcolor='#fff')
        curent_figure.update_traces(mode="markers",  marker=dict(size=4), hovertemplate=None, hoverinfo='skip')
        curent_figure.update_xaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
        curent_figure.update_yaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
        
    else:
        # reconstruct the already created figure as figure from dict
        curent_figure = go.Figure(data = curent_figure['data'], layout = go.Layout(curent_figure['layout']))
        
        # ensure correct values for the sliders
        splitMax = maximum_number_splits
        splitMarks = split_marks
        
        splitpoint = splitpoint_position
        splitPMin = splitpoint_minimum
        splitPMax = splitpoint_max
        
    return dcc.Graph(id='splitpoint_Scatter', figure=curent_figure, config={'displayModeBar': False}), splitMax, splitMarks, splitpoint, splitPMin, splitPMax


#%% main app necessary functions
def app_layout(app, tmpFileNames): 
    """
    Basic interface creation for the interactive applycation. The given inputs let the user manage the application correctly.
    
    Parameters
    ----------
    app : dash.Dash object
        The application we want to create layout on
    
    tmpFileNames : dict
        A dictionary with the names (paths) of the temporary files needed for the execution of the algorithm
    
    """
    app.layout = html.Div(children=[
        dcc.Location(id='url', refresh=False),
        
        html.Div(dcc.Link('X', id='shutdown', href='/shutdown')),
        
        html.H1("dePDDP: Interactive Visualisation"),
        
        html.Ul(children=[
            html.Li(dcc.Link('Clustgering results', href='/clustering_results'), style={'display': 'inline', 'margin': '0px 5px'}),
            html.Li(dcc.Link('Split Point Manipulation', href='/splitpoint_manipulation'), style={'display': 'inline', 'margin': '0px 5px'}),
            # html.Li(dcc.Link('Delete Split', href='/delete_split'), style={'display': 'inline', 'margin': '0px 5px'})
            ],
            style={'margin': '60px 0px 30px 0px'}),
        html.Br(),
        
        html.Div(id='figure_panel'),
        
        dcc.Store(id='cache_dump', data=str(json.dumps(tmpFileNames)))
    ], style={'min-height': '100%', 'width': '900px', 'text-align': 'center', 'margin': 'auto'})


def shutdown():
    """
    Server shutdown function, from visual invironment.
    """
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


#%% home page
def Cluster_Scatter_Plot(object_path):
    """
    Simple scatter plot creation function. This function is used on the initial visualization of the data and can be accessed through out the execution of the interactive visualization server.
    
    Parameters
    ----------
    object_path : str
        The location of the temporary file containg the pickel dump of the dePDDP object we want to visualize
    
    """
    data_matrix, _, _, number_of_nodes = data_preparation(object_path, 0)
    
    # create scatter plot with the splitpoint shape
    category_order = {'cluster': [ str(i) for i in range(len(np.unique(data_matrix['cluster']))) ]}
    color_map = matplotlib.cm.get_cmap('tab20', number_of_nodes)
    colList = { str(i): convert_to_hex(color_map(i)) for i in range(color_map.N) }
    
    # create scatter plot
    figure = px.scatter(data_matrix, x="PC1", y="PC2", color="cluster", category_orders=category_order, color_discrete_map=colList)
    
    # reform visualization
    figure.update_layout(width=850, height=650, plot_bgcolor='#fff')
    figure.update_traces(mode="markers",  marker=dict(size=4), hovertemplate=None, hoverinfo='skip')
    figure.update_xaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
    figure.update_yaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
    
    # Markdown description of the figure
    description = dcc.Markdown('''
            PCA visualization of the data with coloring on the extracted clusters.
            ''')
    
    return html.Div([description, dcc.Graph(figure=figure, config={'displayModeBar': False})])


#%% Split-point manipulation
def Splitpoint_Manipulation(object_path):
    """
    Scatter plot with splitpoint visualization creation function. This function is used on the initial visualization of the data and can be accessed through out the execution of the interactive visualization server.
    
    Parameters
    ----------
    object_path : str
        The location of the temporary file containg the pickel dump of the dePDDP object we want to visualize
    
    """
    data_matrix, splitpoint, internal_nodes, number_of_nodes = data_preparation(object_path, 0)
    
    # create scatter plot with the splitpoint shape
    category_order = {'cluster': [ str(i) for i in np.unique(data_matrix['cluster']) ]}
    color_map = matplotlib.cm.get_cmap('tab20', number_of_nodes)
    colList = { str(i): convert_to_hex(color_map(i)) for i in range(color_map.N) }
    
    # create scatter plot with the splitpoint shape
    figure = px.scatter(data_matrix, x="PC1", y="PC2", color="cluster", hover_name="cluster", category_orders=category_order, color_discrete_map=colList)
    figure.add_shape(type='line', yref="y", xref="x", xsizemode= 'scaled', ysizemode= 'scaled', x0=splitpoint, y0=data_matrix['PC2'].min()*1.2, x1=splitpoint, y1=data_matrix['PC2'].max()*1.2, line=dict(color='red', width=1.5))
    
    # reform visualization
    figure.update_layout(width=850, height=650, plot_bgcolor='#fff')
    figure.update_traces(mode="markers",  marker=dict(size=4), hovertemplate=None, hoverinfo='skip')
    figure.update_xaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
    figure.update_yaxes(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='#aaa', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa')
    
    # Markdown description of the figure
    description = dcc.Markdown('''
            Select the split and the splitpoint in order to manipulate one of the dePDDP`s substeps.
            ''')
    
    # Create the split number slider
    splitStep = html.Div(['Select split: ',dcc.Slider(id='splitView', min=0, max=len(internal_nodes)-1, value=0, marks={str(i): str(i) for i in range(len(internal_nodes))}, step=None)],
                         style={'text-align': 'left', 'width': '740px', 'padding': '20px 80px 20px 80px'})
    
    # Create the main figure of the plot
    scatter = html.Div(id='splitpoint_Main', children=[dcc.Graph(id='splitpoint_Scatter', figure=figure, config={'displayModeBar': False})])
    
    # Create the splitpoint slider
    spSlider = html.Div(['Select splitpoint: ',dcc.Slider(id='splitpoint_Manipulation', min=data_matrix['PC1'].min() - 0.001, max=data_matrix['PC1'].max() + 0.001, step=0.0001, value=splitpoint)], style={'text-align': 'left', 'width': '659px', 'margin': '0px', 'padding': '0px 0px 0px 92px'})
    
    # Create the apply button for the dmanipulation of the splitpoint
    applyButton = html.Button('Apply', id='splitpoint_Man_apply', n_clicks=0)
    
    return html.Div([description, splitStep, scatter, spSlider, applyButton])


#%% Util Functions
def data_preparation(object_path, splitVal):
    """
    Generate the necessary daa for all the visualizations.
    
    Parameters
    ----------
    object_path : str
        The location of the temporary file containg the pickel dump of the dePDDP object we want to visualize
    
    """
    with open(object_path, 'rb') as obj_file:
        tree = pickle.load(obj_file).tree
    
    # Find the each cluster
    clusters = tree.leaves()
    clusters = sorted(clusters, key = lambda x:x.identifier)
    
    root = tree.get_node(0)
    
    # match the points to their respective cluster
    cluster_map = np.zeros(len(root.data['indices']))
    for i in clusters:
        cluster_map[i.data['indices']] = str(int(i.data['color_key']))
    
    # search for the internal nodes (splits) of the tree with the use of the clusters determined above
    dictionary_of_nodes = tree.nodes
    number_of_nodes = len(list(dictionary_of_nodes.keys()))
    leaf_node_list = [ j.identifier for j in clusters ]
    internal_nodes = [ i for i in range(number_of_nodes )if not (i in leaf_node_list) ]
    
    # when nothing din the tree but the root insert the root as internal node (tree theory)
    if len(internal_nodes) == 0: internal_nodes = [0]
    
    # based on the splitVal imported find the node to split 
    node_to_visualize = tree.get_node(internal_nodes[splitVal]) if len(internal_nodes) != 0 else tree.get_node(0)
    
    # create a data matix containg the 1st and 2nd principal components and each clusters respective color key
    data_matrix = pd.DataFrame({
        'PC1': node_to_visualize.data['projection'][:,0],
        'PC2': node_to_visualize.data['projection'][:,1],
        'cluster': cluster_map[node_to_visualize.data['indices']]
        })
    
    data_matrix['cluster'] = data_matrix['cluster'].astype(int).astype(str)
    # determine the splitpoint value
    splitpoint = node_to_visualize.data['splitpoint']
    
    return data_matrix, splitpoint, internal_nodes, number_of_nodes


def convert_to_hex(rgba_color) :
    red = int(rgba_color[0]*255)
    green = int(rgba_color[1]*255)
    blue = int(rgba_color[2]*255)
    return '#{r:02x}{g:02x}{b:02x}'.format(r=red,g=green,b=blue)
    

#%% Manipulation of algorithms results methods
def recalculate_after_spchange(dePDDP_object, split, splitpoint_value):
    """
    Given the serial number of the dePDDP tree`s internal nodes and a new splitpoint value recreate the results of the dePDDP algorithm with the indicated change.
    
    Parameters
    ----------
        dePDDP_object : dePDDP class object
            The object that we want to manipulated on the premiss of this function
            
        split : int
            The serial number of the dePDDP tree`s internal nodes
            
        splitpoint_value : float
            New splitpoint value
        
    Returns
    -------
        self : object
            A dePDDP class type object, with complete results on the algorithm's analysis
    """
    
    tree = dePDDP_object.tree
    # find the cluster nodes
    clusters = tree.leaves()
    clusters = sorted(clusters, key = lambda x:x.identifier)
    
    # find the tree`s internal nodes a.k.a. splits
    dictionary_of_nodes = tree.nodes
    number_of_nodes = len(list(dictionary_of_nodes.keys()))
    leaf_node_list = [ j.identifier for j in clusters ]
    internal_nodes = [ i for i in range(number_of_nodes) if not (i in leaf_node_list) ]
    
    # insure that the root node will be always an internal node while is is the only node in the tree
    if len(internal_nodes) == 0: internal_nodes = [0]
    
    # remove all the splits (nodes) of the tree after the one bing manipulated.
    # this process starts from the last to the first to ensure deletion
    node_keys = list(dictionary_of_nodes.keys())
    for i in range(len(node_keys)-1, 0, -1):
        if node_keys[i] > internal_nodes[split]:
            if tree.get_node(node_keys[i]) != None:
                if not tree.get_node(internal_nodes[split]).is_root():
                    if tree.parent(internal_nodes[split]).identifier != tree.parent(node_keys[i]).identifier:
                        tree.remove_node(node_keys[i])
                else:
                    tree.remove_node(node_keys[i])
    
    # change the split permition of all the internal nodes to True so the algorithm can execute correctly
    dictionary_of_nodes = tree.nodes
    for i in dictionary_of_nodes:
        if dictionary_of_nodes[i].is_leaf():
            if dictionary_of_nodes[i].data['split_density'] != None:
                dictionary_of_nodes[i].data['split_permition'] = True
    
    # reset status variables for the code to execute
    dePDDP_object.node_ids = len(list(dictionary_of_nodes.keys()))-1
    dePDDP_object.cluster_color = len(tree.leaves())+1
    tree.get_node(internal_nodes[split]).data['splitpoint'] = splitpoint_value
    
    # continue the algorithm`s execution from the point left
    dePDDP_object.tree = tree
    dePDDP_object.tree = partial_predict(dePDDP_object)
    
    return dePDDP_object


def partial_predict(dePDDP_object):
    """
    Execute the steps of the algorithm dePDDP untill one of the two stopping creterion is not true.
    
    Parameters
    ----------
        dePDDP_object : dePDDP class object
            The object that we want to manipulated on the premiss of this function
        
    Returns
    -------
        tree : treelib.Tree object
            The newlly created tree after the execution of the algorithm
    """
    tree = dePDDP_object.tree
    
    found_clusters = len(tree.leaves())
    selected_node = dePDDP_object.select_kid(tree.leaves())
    
    while selected_node != None and found_clusters < dePDDP_object.max_clusters_number:  # (ST1) or (ST2)
    
        dePDDP_object.split_function(tree, selected_node)               # step (1)
        
        # select the next kid for split based on the local minimum density
        selected_node = dePDDP_object.select_kid(tree.leaves())   # step (2)
        found_clusters = found_clusters +1                            # (ST1)
        
    return tree













