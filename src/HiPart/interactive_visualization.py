# Copyright (c) 2022 Panagiotis Anagnostou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Interactive visualization module for the algorithms members of the HiPart
package that utilise one decomposition method to one dimension to split the
data.

"""

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from KDEpy import FFTKDE
from HiPart.clustering import DePDDP
from HiPart.clustering import IPDDP
from HiPart.clustering import KMPDDP
from HiPart.clustering import PDDP
from tempfile import NamedTemporaryFile

import dash
import HiPart
import json
import matplotlib
import numpy as np
import os
import pandas as pd
import pickle
import plotly.subplots as subplots
import plotly.express as px
import plotly.graph_objects as go
import signal
import socket
import statsmodels.api as sm
import webbrowser

external_stylesheets = [
    os.path.dirname(HiPart.__file__) + "/assets/int_viz.css"
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
port = -1


def main(inputData):
    """
    The main function of the interactive visualization.

    Parameters
    ----------
    inputData : dePDDP or iPDDP or kM_PDDP or PDDP object
        The object to be visualized.

    Returns (Currently not working correctly)
    ----------------------------------------

    """

    # Compatibility check of the input data with the interactive visualization
    if not (
        isinstance(inputData, DePDDP)
        or isinstance(inputData, IPDDP)
        or isinstance(inputData, KMPDDP)
        or isinstance(inputData, PDDP)
    ):
        raise TypeError(
            """inputData : can only be instances of classes that belong to
            PDDP based algorithm ('dePDDP', 'iPDDP', 'kM-PDDP, PDDP') included
            in HiPart package."""
        )

    if not inputData.visualization_utility:
        raise ValueError(
            """The visualization of the data cannot be achieved because the
            visualization_utility is False."""
        )

    # Create temp files for smooth execution of the interactive visualization
    tmpFilesWrapers = {
        "input_object": NamedTemporaryFile("wb+", suffix=".inv", delete=False),
        "new_input_object": NamedTemporaryFile(
            "wb+",
            suffix=".inv",
            delete=False
        ),
        "figure_path": NamedTemporaryFile("wb+", suffix=".png", delete=False),
    }
    tmpFileNames = {
        "input_object": tmpFilesWrapers["input_object"].name,
        "new_input_object": tmpFilesWrapers["new_input_object"].name,
        "figure_path": tmpFilesWrapers["figure_path"].name,
    }

    # Load the necessary data on the temp files
    with open(tmpFileNames["input_object"], "wb") as input_object_file:
        pickle.dump(inputData, input_object_file)

    with open(tmpFileNames["new_input_object"], "wb") as new_input_object_file:
        pickle.dump(inputData, new_input_object_file)

    # Create and run the server app
    app_layout(app=app, tmpFileNames=tmpFileNames)
    port = _next_free_port(8050)
    webbrowser.open("http://localhost:" + str(port), new=2)
    app.run_server(debug=False, port=port)

    # Recall the object to return before temp file deletion
    with open(tmpFileNames["new_input_object"], "rb") as obj_file:
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


def _next_free_port(port, max_port=65535):
    """
    Find the first port to use after the port given as input.

    Parameters
    ----------
    port : int
        The first port to check.
    max_port : TYPE, optional
        The maximum numbered port to check. The default is 65535.

    Raises
    ------
    IOError
        For no available ports to use.

    Returns
    -------
    port : int
        The number of the first available port.

    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError("no free ports")


# %% Callbacks
@app.callback(
    Output("figure_panel", "children"),
    Input("url", "pathname"),
    State("cache_dump", "data"),
)
def display_menu(pathname, data):
    """
    Menu interchange callback.

    Parameters
    ----------
    pathname : str
        The new url pathname that the webpage needs to go to.
    data : dict
        The paths of the temporary files created for the execution of the
        interactive visualization.

    Returns
    -------
    div : dash.html
        A div containing all the visual components needed for each of the
        page's components.

    """

    # load cashed temp files
    data = json.loads(data)

    # Create the chosen visualization
    if pathname == "/clustering_results":
        div = _Cluster_Scatter_Plot(data["new_input_object"])
    # elif pathname == '/delete_split':
    #    div = Delete_Split_Plot(data['new_input_object'], data['figure_path'])
    elif pathname == "/splitpoint_manipulation":
        div = _Splitpoint_Manipulation(data["new_input_object"])
    elif pathname == "/shutdown":
        div = html.Div(children=[html.Br(), html.H4("Close the Window!!!")])
        _shutdown()
    else:
        div = _Cluster_Scatter_Plot(data["input_object"])

        # Reset the visualization manipulations
        with open(data["input_object"], "rb") as obj_file:
            obj = pickle.load(obj_file)

        with open(data["new_input_object"], "wb") as obj_file:
            pickle.dump(obj, obj_file)

    return div


@app.callback(
    Output("splitpoint_Main", "children"),
    Output("splitView", "max"),
    Output("splitView", "marks"),
    Output("splitpoint_Manipulation", "value"),
    Output("splitpoint_Manipulation", "min"),
    Output("splitpoint_Manipulation", "max"),
    Output("splitpoint_Manipulation", "marks"),
    State("cache_dump", "data"),
    State("splitpoint_Scatter", "figure"),
    Input("splitView", "value"),
    State("splitView", "max"),
    State("splitView", "marks"),
    Input("splitpoint_Manipulation", "value"),
    State("splitpoint_Manipulation", "min"),
    State("splitpoint_Manipulation", "max"),
    State("splitpoint_Manipulation", "marks"),
    Input("splitpoint_Man_apply", "n_clicks"),
    prevent_initial_call=True,
)
def Splitpoint_Manipulation_Callback(
    data,
    current_figure,
    split_number,
    maximum_number_splits,
    split_marks,
    splitpoint_position,
    splitpoint_minimum,
    splitpoint_max,
    splitpoint_marks,
    apply_button,
):
    """
    Function triggered by the change in value on the Input elements. The
    triggers are:

    1. the change in the split of the data.
    2. the changes in the split-point.
    3. the apply button for the change to the new split-point (partial
       algorithm execution from the selected split).

    The rest of the function's inputs are inputs that can't trigger this
    callback but their data are necessary for the execution of this callback.

    Parameters
    ----------
    splitpoint_marks
    data : dict
        The paths of the temporary files created for the execution of the
        interactive visualization.
    current_figure : dict
        A dictionary created from the plotly express object plots. It is used
        for the manipulation of the current figure.
    split_number : int
        The value of the split to project (or is projected, depending on the
        callback context triggered state).
    maximum_number_splits : int
        The number of splits there are in the model created/manipulated.
    split_marks : dict
        The split number is the key of the dictionary and the assigned value
        is the value of the dictionary.
    splitpoint_position : float
        The current position of the shape represents the split-point of the
        currently selected split, extracted from the split-point slider.
    splitpoint_minimum : float
        The minimum value that can be assigned to the split-point, extracted
        from the split-point slider.
    splitpoint_max : float
        The maximum value that can be assigned to the split-point, extracted
        from the split-point slider.
    apply_button : int
        The number of clicks of the apply button (Not needed in the function`s
        execution, but necessary for the callback definition).

    Returns
    -------
    figure : dash.dcc.Graph
        A figure that can be integrated at dash`s Html components.
    splitMax : float
        The new value for the maximum split number.
    splitMarks : float
        The changed marks the spit can utilize as values.
    splitpoint : float
        The newly generated split-point by the callback.
    splitPMin : int
        The minimum value the split-point can take.
    splitPMax : int
        The maximum value the split-point can take.

    """

    callback_context = dash.callback_context
    callback_ID = callback_context.triggered[0]["prop_id"].split(".")[0]

    # load cashed temp files
    data = json.loads(data)

    # Basic check on the callback trigger, based on the dash html elements ID.
    if callback_ID == "splitView":
        (
            data_matrix,
            splitpoint,
            internal_nodes,
            number_of_nodes,
        ) = _data_preparation(
            data["new_input_object"], split_number
        )

        # ensure correct values for the sliders
        splitPMarks = {
            splitpoint: {
                'label': 'Generated Split-point',
                'style': {'color': '#77b0b1'}
            }
        }

        splitMax = maximum_number_splits
        splitMarks = split_marks

        splitPMin = data_matrix["PC1"].min()
        splitPMax = data_matrix["PC1"].max()

        # create visualization points
        category_order = {
            "cluster": [
                str(i) for i in range(len(np.unique(data_matrix["cluster"])))
            ]
        }
        color_map = matplotlib.cm.get_cmap("tab20", number_of_nodes)
        colList = {str(i): _convert_to_hex(color_map(i)) for i in range(color_map.N)}

        with open(data["new_input_object"], "rb") as cls_file:
            cls = pickle.load(cls_file)

        # Check data compatibility with the function
        if isinstance(cls, DePDDP):
            current_figure = _int_make_scatter_n_hist(
                data_matrix,
                splitpoint,
                cls.bandwidth_scale,
                category_order,
                colList,
            )
        else:
            current_figure = _int_make_scatter_n_marginal_scatter(
                data_matrix,
                splitpoint,
                category_order,
                colList
            )

    elif callback_ID == "splitpoint_Manipulation":
        # reconstruct the already created figure as figure from dict
        current_figure = go.Figure(
            data=current_figure["data"],
            layout=go.Layout(current_figure["layout"]),
        )

        # update the splitpoint shape location
        current_figure.update_shapes(
            {"x0": splitpoint_position, "x1": splitpoint_position}
        )

        # ensure correct values for the sliders
        splitMax = maximum_number_splits
        splitMarks = split_marks

        splitpoint = splitpoint_position
        splitPMin = splitpoint_minimum
        splitPMax = splitpoint_max
        splitPMarks = splitpoint_marks

    elif callback_ID == "splitpoint_Man_apply":
        # execution of the split-point manipulation algorithmically
        with open(data["new_input_object"], "rb") as cls_file:
            cls = pickle.load(cls_file)

        cls = _recalculate_after_spchange(
            cls,
            split_number,
            splitpoint_position
        )

        with open(data["new_input_object"], "wb") as cls_file:
            pickle.dump(cls, cls_file)

        # reconstruction of the figure and its slider from scratch
        data_matrix, splitpoint, internal_nodes, number_of_nodes = _data_preparation(
            data["new_input_object"], split_number
        )

        # ensure correct values for the sliders
        splitMax = len(internal_nodes) - 1
        splitMarks = {str(i): str(i) for i in range(len(internal_nodes))}

        splitPMin = data_matrix["PC1"].min()
        splitPMax = data_matrix["PC1"].max()
        splitPMarks = {
            splitpoint: {
                'label': 'Generated Split-point',
                'style': {'color': '#77b0b1'}
            }
        }

        # create visualization points
        order = {
            "cluster": [
                str(i) for i in range(len(np.unique(data_matrix["cluster"])))
            ]
        }
        map = matplotlib.cm.get_cmap("tab20", number_of_nodes)
        color_list = {str(i): _convert_to_hex(map(i)) for i in range(map.N)}

        with open(data["new_input_object"], "rb") as obj_file:
            obj = pickle.load(obj_file)

        # Check data compatibility with the function
        if isinstance(obj, DePDDP):
            current_figure = _int_make_scatter_n_hist(
                data_matrix,
                splitpoint,
                obj.bandwidth_scale,
                order,
                color_list,
            )
        else:
            current_figure = _int_make_scatter_n_marginal_scatter(
                data_matrix,
                splitpoint,
                order,
                color_list
            )

    else:
        # reconstruct the already created figure as figure from dict
        current_figure = go.Figure(
            data=current_figure["data"],
            layout=go.Layout(current_figure["layout"])
        )

        # ensure correct values for the sliders
        splitMax = maximum_number_splits
        splitMarks = split_marks

        splitpoint = splitpoint_position
        splitPMin = splitpoint_minimum
        splitPMax = splitpoint_max
        splitPMarks = splitpoint_marks

    return (
        dcc.Graph(
            id="splitpoint_Scatter",
            figure=current_figure,
            config={"displayModeBar": False},
        ),
        splitMax,
        splitMarks,
        splitpoint,
        splitPMin,
        splitPMax,
        splitPMarks,
    )


# %% Split-point Manipulation page
def _Splitpoint_Manipulation(object_path):
    """
    Scatter plot with split-point visualization creation function. This
    function is used on the initial visualization of the data and can be
    accessed throughout the execution of the interactive visualization server.

    Parameters
    ----------
    object_path : str
        The location of the temporary file containing the pickle dump of the
        object we want to visualize.

    """

    # get the necessary data for the visualization
    (
        data_matrix,
        splitpoint,
        internal_nodes,
        number_of_nodes,
    ) = _data_preparation(
        object_path, 0
    )

    # create an ordering for the legend of the visualization
    category_order = {
        "cluster": [str(i) for i in np.unique(data_matrix["cluster"])]
    }

    # generate the colors to be used
    color_map = matplotlib.cm.get_cmap("tab20", number_of_nodes)
    colList = {
        str(i): _convert_to_hex(color_map(i)) for i in range(color_map.N)
    }

    with open(object_path, "rb") as obj_file:
        obj = pickle.load(obj_file)

    # Check data compatibility with the function
    if isinstance(obj, DePDDP):
        figure = _int_make_scatter_n_hist(
            data_matrix,
            splitpoint,
            obj.bandwidth_scale,
            category_order,
            colList,
        )
    else:
        figure = _int_make_scatter_n_marginal_scatter(
            data_matrix,
            splitpoint,
            category_order,
            colList,
        )

    # Markdown description of the figure
    description = dcc.Markdown(
        _message_center("des:splitpoitn_man", object_path),
        style={
            "text-align": "left",
            "margin": "-20px 0px 0px 0px",
        }
    )

    # Create the split number slider
    splitStep = html.Div(
        [
            "Select split: ",
            dcc.Slider(
                id="splitView",
                min=0,
                max=len(internal_nodes) - 1,
                value=0,
                marks={str(i): str(i) for i in range(len(internal_nodes))},
                step=1,
            ),
        ],
        style={
            "text-align": "left",
            "width": "740px",
            "padding": "20px 80px 20px 80px",
        },
    )

    # Create the main figure of the plot
    scatter = html.Div(
        id="splitpoint_Main",
        children=[
            dcc.Graph(
                id="splitpoint_Scatter",
                figure=figure,
                config={"displayModeBar": False, "autosizable": True}
            )
        ],
        style={"margin": "-20px 0px -30px 0px"},
    )

    # Create the split-point slider
    splitSlider = html.Div(
        [
            "Select splitpoint: ",
            dcc.Slider(
                id="splitpoint_Manipulation",
                min=data_matrix["PC1"].min() - 0.001,
                max=data_matrix["PC1"].max() + 0.001,
                marks={
                    splitpoint: {
                        'label': 'Generated Splitpoint',
                        'style': {'color': '#77b0b1'}
                    }
                },
                step=0.001,
                value=splitpoint,
                included=False,
            ),
        ],
        style={
            "text-align": "left",
            "width": "627px",
            "margin": "0px",
            "padding": "25px 0px 0px 112px",
        },
    )

    # Create the apply button for the manipulation of the split-point
    applyButton = html.Button("Apply", id="splitpoint_Man_apply", n_clicks=0)

    return html.Div([
        description,
        splitStep,
        scatter,
        splitSlider,
        applyButton,
    ])


# %% Utilities of the interactive visualization
def _data_preparation(object_path, splitVal):
    """
    Generate the necessary data for all the visualizations.

    Parameters
    ----------
    object_path : str
        The location of the temporary file containing the pickle dump of the
        object we want to visualize.
    splitVal : int
        The serial number of split that want to extract data from.

    Returns
    -------
    data_matrix : pandas.core.frame.DataFrame
    splitpoint : int
    internal_nodes : list
    number_of_nodes : int

    """

    # load the data from the temp files
    with open(object_path, "rb") as obj_file:
        tree = pickle.load(obj_file).tree

    # Find the clusters from the algorithms tree
    clusters = tree.leaves()
    clusters = sorted(clusters, key=lambda x: x.identifier)

    root = tree.get_node(0)

    # match the points to their respective cluster
    cluster_map = np.zeros(len(root.data["indices"]))
    for i in clusters:
        cluster_map[i.data["indices"]] = str(int(i.data["color_key"]))

    # list of all the tree's nodes
    node_dictionary = tree.nodes

    # Search for the internal nodes (splits) of the tree with the use of the
    # clusters determined above
    number_of_nodes = len(list(node_dictionary.keys()))
    leaf_node_list = [j.identifier for j in clusters]
    internal_nodes = [
        i for i in range(number_of_nodes) if not (i in leaf_node_list)
    ]

    # When nothing din the tree but the root insert the root as internal
    # node (tree theory)
    if len(internal_nodes) == 0:
        internal_nodes = [0]

    # based on the splitVal imported find the node to split
    node_to_visualize = (
        tree.get_node(internal_nodes[splitVal])
        if len(internal_nodes) != 0
        else tree.get_node(0)
    )

    # create a data matrix containing the 1st and 2nd principal components and
    # each clusters respective color key
    data_matrix = pd.DataFrame(
        {
            "PC1": node_to_visualize.data["projection"][:, 0],
            "PC2": node_to_visualize.data["projection"][:, 1],
            "cluster": cluster_map[node_to_visualize.data["indices"]],
        }
    )

    data_matrix["cluster"] = data_matrix["cluster"].astype(int).astype(str)

    # determine the split-point value
    splitpoint = node_to_visualize.data["splitpoint"]

    return data_matrix, splitpoint, internal_nodes, number_of_nodes


def _convert_to_hex(rgba_color):
    """
    Convert the color encoding from RGBa to hexadecimal for integration with
    the CSS.

    Parameters
    ----------
    rgba_color : tuple
        A tuple of floats containing the RGBa values.

    Returns
    -------
    str
        The hexadecimal value of the color in question.

    """

    red = int(rgba_color[0] * 255)
    green = int(rgba_color[1] * 255)
    blue = int(rgba_color[2] * 255)

    return "#{r:02x}{g:02x}{b:02x}".format(r=red, g=green, b=blue)


def _message_center(message_id, object_path):
    """
    This function is responsible for saving and returning all the messages seen
    in the interactive visualization.

    Parameters
    ----------
    object_path : TYPE
        The absolute path of the object the interactive visualization
        visualizes.
    message_id : TYPE
        The id of the message to be returned.

    Returns
    -------
    message : str
        The requested message.

    """

    with open(object_path, "rb") as obj_file:
        obj = pickle.load(obj_file)

    obj_name = str(obj.__class__).split(".")[-1].split("'")[0]

    msg = ""

    if message_id == "des:main_cluser":
        msg = """
        ### """ + obj_name + """: Basic visualization

        This is the basic visualization for the clustering of the input data.
        This figure is generated by visualizing the first two components of the
        decomposition method used by the """ + obj_name + """ algorithm. The
        colors on visualization represent each of the separate clusters
        extracted by the execution of the """ + obj_name + """ algorithm. It is
        important to notice that each color represents the same cluster
        throughout the execution of the interactive visualization (the colors
        on the clusters only change with the manipulation of the execution of
        the algorithm).

        """

    elif message_id == "des:splitpoitn_man":
        msg = """
        ### """ + obj_name + """: Splitpoint Manipulation

        On this page, you can manipulate the split point of the
        """ + obj_name + """ algorithm. The process is similar to all the
        algorithms, members of the HiPart package.

        For the split-point manipulation, the top section of the figure is the
        selection of split to manipulate. This can be done with the help of the
        top sliding bar. The numbers below the bar each time represent the
        serial number of the split to manipulate. Zero is the first split of
        the data set. It is necessary to notice that the manipulation of the
        split point because of the nature of the execution must start from the
        earliest to the latest split otherwise, the manipulation will be lost.

        The middle section of the figure visualizes the data with the
        utilization of the decomposition technique used to execute the
        """ + obj_name + """ algorithm. The colors on the scatter plot
        represent the final clusters of the input dataset. The red vertical
        line represents the split-point of the data for the current split of
        the algorithm. """

        if obj_name == "dePDDP":
            msg += """The marginal plot for the x-axis of this visualization
            represents the density of the data. The reason behind that is to
            visualize the information the algorithm has to split the data.

            """
        else:
            msg += """The marginal plot for the x-axis of this visualization is
            one dimension scatter plot of the split, the data of which we
            visualize. The reason behind that is to visualize the information
            the algorithm has to split the data.

            """
        msg += """
        Finally, the bottom section of the figure allows the manipulation of
        the split-point with the utilization of a continuous sliding bar. By
        changing positions on the node of the sliding-bar we can see
        corresponding movements on the vertical red line that represents the
        split-point of the split.

        The apply button, at the bottom of the page, applies the manipulated
        split-point and re-executes the algorithm for the rest of the splits
        that appeared after the currently selected one.
        """

    return msg


# %% app construction page
def app_layout(app, tmpFileNames):
    """
    Basic interface creation for the interactive application. The given inputs
    let the user manage the application correctly.

    Parameters
    ----------
    app : dash.Dash
        The application we want to create the layout on.
    tmpFileNames : dict
        A dictionary with the names (paths) of the temporary files needed for
        the execution of the algorithm.

    """

    app.layout = html.Div(
        children=[
            # ----------------------------------------------------------------
            # Head of the interactive visualization
            dcc.Location(id="url", refresh=False),
            html.Div(dcc.Link("X", id="shutdown", href="/shutdown")),
            html.H1("HiPart: Interactive Visualisation"),
            html.Ul(
                children=[
                    html.Li(
                        dcc.Link(
                            "Clustering results",
                            href="/clustering_results"
                        ),
                        style={"display": "inline", "margin": "0px 5px"},
                    ),
                    html.Li(
                        dcc.Link(
                            "Split Point Manipulation",
                            href="/splitpoint_manipulation"
                        ),
                        style={"display": "inline", "margin": "0px 5px"},
                    ),
                    # html.Li(dcc.Link('Delete Split', href='/delete_split'),
                    #         style={'display': 'inline', 'margin': '0px 5px'})
                ],
                style={"margin": "60px 0px 30px 0px"},
            ),
            html.Br(),
            # ----------------------------------------------------------------
            # Main section of the interactive visualization
            html.Div(id="figure_panel"),
            # ----------------------------------------------------------------
            # cached data container, local on the browser
            dcc.Store(id="cache_dump", data=str(json.dumps(tmpFileNames))),
        ],
        style={
            "min-height": "100%",
            "width": "900px",
            "text-align": "center",
            "margin": "auto",
        },
    )


def _shutdown():
    """
    Server shutdown function, from visual environment.
    """
    # func = request.environ.get("werkzeug.server.shutdown")
    # if func is None:
    #     raise RuntimeError("Not running with the Werkzeug Server")
    # func()
    os.kill(os.getpid(), signal.SIGINT)


def _Cluster_Scatter_Plot(object_path):
    """
    Simple scatter plot creation function. This function is used on the
    initial visualization of the data and can be accessed throughout the
    execution of the interactive visualization server.

    Parameters
    ----------
    object_path : str
        The location of the temporary file containing the pickle dump of the
        object we want to visualize.

    """

    # get the necessary data for the visualization
    (
        data_matrix,
        _,
        _,
        number_of_nodes,
    ) = _data_preparation(object_path, 0)

    # create scatter plot with the split-point shape
    color_map = matplotlib.cm.get_cmap("tab20", number_of_nodes)
    colList = {
        str(i): _convert_to_hex(color_map(i)) for i in range(color_map.N)
    }
    category_order = {
        "cluster": [
            str(i) for i in range(len(np.unique(data_matrix["cluster"])))
        ]
    }

    # create scatter plot
    figure = px.scatter(
        data_matrix,
        x="PC1",
        y="PC2",
        color="cluster",
        category_orders=category_order,
        color_discrete_map=colList,
    )

    # reform visualization
    figure.update_layout(width=850, height=650, plot_bgcolor="#fff")
    figure.update_traces(
        mode="markers",
        marker=dict(size=4),
        hovertemplate=None,
        hoverinfo="skip"
    )
    figure.update_xaxes(
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaa",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#aaa",
    )
    figure.update_yaxes(
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaa",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#aaa",
    )

    # Markdown description of the figure
    description = dcc.Markdown(
        _message_center("des:main_cluser", object_path),
        style={
            "text-align": "left",
            "margin": "-20px 0px 0px 0px",
        }
    )

    return html.Div(
        [
            description,
            dcc.Graph(figure=figure, config={"displayModeBar": False})
        ]
    )


# %% Basic plots
def _int_make_scatter_n_hist(
        data_matrix,
        splitPoint,
        bandwidth_scale,
        category_order,
        colList
):
    """
    Create two plots that visualize the data on the second plot and on the
    first give their density representation on the first principal component.

    Parameters
    ----------
    data_matrix : pandas.core.frame.DataFrame
        The projection of the data on the first two Principal Components as
        columns "PC1" and "PC2" and the final cluster each sample belong at
        the end of the algorithm's execution as column "cluster".
    splitPoint : int
        The values of the point the data are split for this plot.
    bandwidth_scale
        Standard deviation scaler for the density approximation. Allowed values
        are in the (0,1).
    category_order : dict
        The order of witch to show the clusters, contained in the
        visualization, on the legend of the plot.
    colList : dict
        A dictionary containing the color of each cluster (key) as RGBa tuple
        (value).

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The reulted figure of the function.

    """

    bandwidth = sm.nonparametric.bandwidths.select_bandwidth(
        data_matrix["PC1"],
        "silverman",
        kernel=None
    )
    s, e = FFTKDE(
        kernel="gaussian",
        bw=(bandwidth_scale * bandwidth)
    ).fit(data_matrix["PC1"].to_numpy()).evaluate()

    fig = subplots.make_subplots(
        rows=2, cols=1,
        row_heights=[0.15, 0.85],
        vertical_spacing=0.02,
        shared_yaxes=False,
        shared_xaxes=True,
    )

    fig.add_trace(go.Scatter(
        x=s, y=e,
        mode="lines",
        line=dict(color='royalblue', width=1.3),
        name='PC1',
        hovertemplate=None,
        hoverinfo="skip",
        showlegend=False
    ), row=1, col=1)
    fig.add_shape(
        type="line",
        yref="y", xref="x",
        xsizemode="scaled",
        ysizemode="scaled",
        x0=splitPoint,
        x1=splitPoint,
        y0=-0.005,
        y1=e.max()*1.2,
        line=dict(color="red", width=1.5),
        row=1, col=1,
    )

    main_figure = px.scatter(
        data_matrix,
        x="PC1", y="PC2",
        color="cluster",
        hover_name="cluster",
        category_orders=category_order,
        color_discrete_map=colList,
    )["data"]
    for i in main_figure:
        fig.add_trace(i, row=2, col=1)
    fig.update_traces(
        mode="markers",
        marker=dict(size=4),
        hovertemplate=None,
        hoverinfo="skip",
        row=2, col=1,
    )
    fig.add_shape(
        type="line",
        yref="y", xref="x",
        xsizemode="scaled",
        ysizemode="scaled",
        x0=splitPoint,
        x1=splitPoint,
        y0=data_matrix["PC2"].min() * 1.2,
        y1=data_matrix["PC2"].max() * 1.2,
        line=dict(color="red", width=1.5),
        row=2, col=1,
    )

    # reform visualization
    fig.update_layout(
        width=850, height=700,
        plot_bgcolor="#fff",
        margin={"t": 20, "b": 50},
    )
    fig.update_xaxes(
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaa",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#aaa",
    )
    fig.update_yaxes(
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaa",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#aaa",
    )

    return fig


def _int_make_scatter_n_marginal_scatter(
        data_matrix,
        splitPoint,
        category_order,
        colList,
        centers=None
):
    """
    Create two plots that visualize the data on the second plot and on the
    first give their presentation on the first principal component.

    Parameters
    ----------
    data_matrix : pandas.core.frame.DataFrame
        The projection of the data on the first two Principal Components as
        columns "PC1" and "PC2" and the final cluster each sample belong at
        the end of the algorithm's execution as column "cluster".
    splitPoint : int
        The values of the point the data are split for this plot.
    category_order : dict
        The order of witch to show the clusters, contained in the
        visualization, on the legend of the plot.
    colList : dict
        A dictionary containing the color of each cluster (key) as RGBa tuple
        (value).
    centers : numpy.ndarray
        The values of the k-means' centers for the clustering of the data
        projected on the first principal component for each split, for the
        kM-PDDP algorithm.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The reulted figure of the function.

    """

    fig = subplots.make_subplots(
        rows=2, cols=1,
        row_heights=[0.15, 0.85],
        vertical_spacing=0.02,
        shared_yaxes=False,
        shared_xaxes=True,
    )

    # Create the marginal scatter plot of the figure (projection on one
    # principal component)
    marginal_figure = px.scatter(
        data_matrix,
        x="PC1",
        y=np.zeros(data_matrix["PC1"].shape[0]),
        color="cluster",
        hover_name="cluster",
        category_orders=category_order,
        color_discrete_map=colList,
    )["data"]
    for i in marginal_figure:
        fig.add_trace(i, row=1, col=1)
    fig.update_traces(
        mode="markers",
        marker=dict(size=5),
        hovertemplate=None,
        hoverinfo="skip",
        showlegend=False,
        row=1, col=1,
    )
    # If there are k-Means centers add them on the marginal scatter plot
    if centers is not None:
        fig.add_trace(go.Scatter(
            x=centers,
            y=np.zeros(2),
            mode="markers",
            marker=dict(symbol=22, color='darkblue', size=15),
            name='centers',
            hovertemplate=None,
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=1)
    fig.add_shape(
        type="line",
        yref="y", xref="x",
        xsizemode="scaled",
        ysizemode="scaled",
        x0=splitPoint,
        x1=splitPoint,
        y0=-2.5, y1=2.5,
        line=dict(color="red", width=1.5),
        row=1, col=1,
    )

    # Create the main scatter plot of the figure
    main_figure = px.scatter(
        data_matrix,
        x="PC1", y="PC2",
        color="cluster",
        hover_name="cluster",
        category_orders=category_order,
        color_discrete_map=colList,
    )["data"]
    for i in main_figure:
        fig.add_trace(i, row=2, col=1)
    fig.update_traces(
        mode="markers", marker=dict(size=4),
        hovertemplate=None, hoverinfo="skip",
        row=2, col=1,
    )
    fig.add_shape(
        type="line",
        yref="y", xref="x",
        xsizemode="scaled",
        ysizemode="scaled",
        x0=splitPoint,
        x1=splitPoint,
        y0=data_matrix["PC2"].min() * 1.2,
        y1=data_matrix["PC2"].max() * 1.2,
        line=dict(color="red", width=1.5),
        row=2, col=1,
    )

    # Reform visualization
    fig.update_layout(
        width=850,
        height=700,
        plot_bgcolor="#fff",
        margin={"t": 20, "b": 50}
    )
    fig.update_xaxes(
        fixedrange=True, showgrid=True,
        gridwidth=1, gridcolor="#aaa",
        zeroline=True, zerolinewidth=1,
        zerolinecolor="#aaa"
    )
    fig.update_yaxes(
        fixedrange=True, showgrid=True,
        gridwidth=1, gridcolor="#aaa",
        zeroline=True, zerolinewidth=1,
        zerolinecolor="#aaa"
    )

    return fig


# %% Algorithm re-execution tasks
def _recalculate_after_spchange(hipart_object, split, splitpoint_value):
    """
    Given the serial number of the HiPart algorithm tree`s internal nodes and a
    new split-point value recreate the results of the HiPart member algorithm
    with the indicated change.

    Parameters
    ----------
    hipart_object : dePDDP or iPDDP or kM_PDDP or PDDP object
        The object that we want to manipulate on the premiss of this function.
    split : int
        The serial number of the dePDDP tree`s internal nodes.
    splitpoint_value : float
        New split-point value.

    Returns
    -------
    hipart_object : dePDDP or iPDDP or kM_PDDP or PDDP object
        A dePDDP class type object, with complete results on the algorithm's
        analysis

    """

    tree = hipart_object.tree

    # find the cluster nodes
    clusters = tree.leaves()
    clusters = sorted(clusters, key=lambda x: x.identifier)

    # find the tree`s internal nodes a.k.a. splits
    dictionary_of_nodes = tree.nodes
    number_of_nodes = len(list(dictionary_of_nodes.keys()))
    leaf_node_list = [j.identifier for j in clusters]
    internal_nodes = [
        i for i in range(number_of_nodes) if not (i in leaf_node_list)
    ]

    # Insure that the root node will be always an internal node while is the
    # only node in the tree
    if len(internal_nodes) == 0:
        internal_nodes = [0]

    # remove all the splits (nodes) of the tree after the one bing manipulated.
    # this process starts from the last to the first to ensure deletion
    node_keys = list(dictionary_of_nodes.keys())
    for i in range(len(node_keys) - 1, 0, -1):
        if node_keys[i] > internal_nodes[split]:
            if tree.get_node(node_keys[i]) is not None:
                if not tree.get_node(internal_nodes[split]).is_root():
                    if (
                        tree.parent(internal_nodes[split]).identifier
                        != tree.parent(node_keys[i]).identifier
                    ):
                        tree.remove_node(node_keys[i])
                else:
                    tree.remove_node(node_keys[i])

    # change the split permission of all the internal nodes to True so the
    # algorithm can execute correctly
    dictionary_of_nodes = tree.nodes
    for i in dictionary_of_nodes:
        if dictionary_of_nodes[i].is_leaf():
            if dictionary_of_nodes[i].data["split_criterion"] is not None:
                dictionary_of_nodes[i].data["split_permission"] = True

    # reset status variables for the code to execute
    hipart_object.node_ids = len(list(dictionary_of_nodes.keys())) - 1
    hipart_object.cluster_color = len(tree.leaves()) + 1
    tree.get_node(internal_nodes[split]).data["splitpoint"] = splitpoint_value

    # continue the algorithm`s execution from the point left
    hipart_object.tree = tree
    hipart_object.tree = _partial_predict(hipart_object)

    return hipart_object


def _partial_predict(hipart_object):
    """
    Execute the steps of the algorithm dePDDP until one of the two stopping
    criterion is not true.

    Parameters
    ----------
    hipart_object : dePDDP or iPDDP or kM_PDDP or PDDP object
        The object that we want to manipulate on the premiss of this function.

    Returns
    -------
    tree : treelib.Tree
        The newly created tree after the execution of the algorithm.

    """

    tree = hipart_object.tree

    found_clusters = len(tree.leaves())
    selected_node = hipart_object.select_kid(tree.leaves())

    while (
        (selected_node is not None)
        and (found_clusters < hipart_object.max_clusters_number)
    ):  # (ST1) or (ST2)

        hipart_object.split_function(tree, selected_node)  # step (1)

        # select the next kid for split based on the local minimum density
        selected_node = hipart_object.select_kid(tree.leaves())  # step (2)
        found_clusters = found_clusters + 1  # (ST1)

    return tree
