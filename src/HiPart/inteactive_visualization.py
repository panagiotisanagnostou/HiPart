# -*- coding: utf-8 -*-
"""
Interactive visualization module for the algorithms members of the HiPart
package that utilise one decomposition method to one dimention to split the
data.

"""

import dash
import HiPart
import HiPart.__utility_functions as util
import json
import matplotlib
import numpy as np
import os
import pickle
import plotly.graph_objects as go
import socket
import webbrowser

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from HiPart.clustering import dePDDP
from HiPart.clustering import iPDDP
from HiPart.clustering import kM_PDDP
from HiPart.clustering import PDDP
from tempfile import NamedTemporaryFile

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

    Returns (Curently not working correclty)
    ----------------------------------------
    obj : dePDDP or iPDDP or kM_PDDP or PDDP object
        The manipulated by the interactive visualization HiPart object. The
        object`s type depends on the input object.

    """

    # Compatibillity check of the input data with the interactive visualization
    if not (
        isinstance(inputData, dePDDP)
        or isinstance(inputData, iPDDP)
        or isinstance(inputData, kM_PDDP)
        or isinstance(inputData, PDDP)
    ):
        raise TypeError(
            """inputData : can only be instances of classes that belong to
            PDDP based algoirthm ('dePDDP', 'iPDDP', 'kM-PDDP, PDDP') included
            in HiPart package."""
        )

    if not inputData.visualization_utility:
        raise ValueError(
            """The visulaization of the data cannot be achived because the
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
    util.app_layout(app=app, tmpFileNames=tmpFileNames)
    port = next_free_port(8050)
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


def next_free_port(port, max_port=65535):
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
        div = util.Cluster_Scatter_Plot(data["new_input_object"])
    # elif pathname == '/delete_split':
    #    div = Delete_Split_Plot(data['new_input_object'], data['figure_path'])
    elif pathname == "/splitpoint_manipulation":
        div = Splitpoint_Manipulation(data["new_input_object"])
    elif pathname == "/shutdown":
        div = html.Div(children=[html.Br(), html.H4("Close the Window!!!")])
        util.shutdown()
    else:
        div = util.Cluster_Scatter_Plot(data["input_object"])

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
    State("cache_dump", "data"),
    State("splitpoint_Scatter", "figure"),
    Input("splitView", "value"),
    State("splitView", "max"),
    State("splitView", "marks"),
    Input("splitpoint_Manipulation", "value"),
    State("splitpoint_Manipulation", "min"),
    State("splitpoint_Manipulation", "max"),
    Input("splitpoint_Man_apply", "n_clicks"),
    prevent_initial_call=True,
)
def Splitpoint_Manipulation_Callback(
    data,
    curent_figure,
    split_number,
    maximum_number_splits,
    split_marks,
    splitpoint_position,
    splitpoint_minimum,
    splitpoint_max,
    apply_button,
):
    """
    Function triggered by the change in value on the Input elements. The
    triggers are:

    1. the change in the split of the data.
    2. the changes in the splitpoint.
    3. the apply button for the change to the new splitpoint (partial
       algorithm execution from the selected split).

    The rest of the function's inputs are inputs that can't trigger this
    callback but their data are necessary for the execution of this callback.

    Parameters
    ----------
    data : dict
        The paths of the temporary files created for the execution of the
        interactive visualization.
    curent_figure : dict
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
        The current position of the shape represents the splitpoint of the
        currently selected split, extracted from the splitpoint slider.
    splitpoint_minimum : float
        The minimum value that can be assigned to the splitpoint, extracted
        from the splitpoint slider.
    splitpoint_max : float
        The maximum value that can be assigned to the splitpoint, extracted
        from the splitpoint slider.
    apply_button : int
        The number of clicks of the apply button (Not needed in the function`s
        execution, but necessary for the callback definition).

    Returns
    -------
    figure : dash.dcc.Graph
        A figure that can be intergraded at dash`s Html components.
    splitMax : float
        The new value for the maximum split number.
    splitMarks : float
        The changed marks the spit can utilize as values.
    splitpoint : float
        The newly generated splitpoint by the callback.
    splitPMin : int
        The minimum value the splitpoint can take.
    splitPMax : int
        The maximum value the splitpoint can take.

    """

    callback_context = dash.callback_context
    callback_ID = callback_context.triggered[0]["prop_id"].split(".")[0]

    # load cashed temp files
    data = json.loads(data)

    # Basic check on the callback triger, based on the dash html elements ID.
    if callback_ID == "splitView":
        data_matrix, splitpoint, internal_nodes, number_of_nodes = util.data_preparation(
            data["new_input_object"], split_number
        )

        # ensure correct values for the sliders
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
        colList = {str(i): util.convert_to_hex(color_map(i)) for i in range(color_map.N)}

        with open(data["new_input_object"], "rb") as obj_file:
            obj = pickle.load(obj_file)

        # Check data compatibility with the function
        if isinstance(obj, dePDDP):
            curent_figure = util.int_make_scatter_n_hist(
                data_matrix,
                splitpoint,
                obj.split_data_bandwidth_scale,
                category_order,
                colList,
            )
        else:
            curent_figure = util.int_make_scatter_n_marginal_scatter(
                data_matrix,
                splitpoint,
                category_order,
                colList
            )

    elif callback_ID == "splitpoint_Manipulation":
        # reconstruct the already created figure as figure from dict
        curent_figure = go.Figure(
            data=curent_figure["data"],
            layout=go.Layout(curent_figure["layout"]),
        )

        # update the splitpoint shape location
        curent_figure.update_shapes(
            {"x0": splitpoint_position, "x1": splitpoint_position}
        )

        # ensure correct values for the sliders
        splitMax = maximum_number_splits
        splitMarks = split_marks

        splitpoint = splitpoint_position
        splitPMin = splitpoint_minimum
        splitPMax = splitpoint_max

    elif callback_ID == "splitpoint_Man_apply":
        # execution of the splitpoint manipulation algorithmically
        with open(data["new_input_object"], "rb") as obj_file:
            obj = pickle.load(obj_file)

        obj = util.recalculate_after_spchange(
            obj,
            split_number,
            splitpoint_position
        )

        with open(data["new_input_object"], "wb") as obj_file:
            pickle.dump(obj, obj_file)

        # reconstrauction of the figure and its slider from scrach
        data_matrix, splitpoint, internal_nodes, number_of_nodes = util.data_preparation(
            data["new_input_object"], split_number
        )

        # ensure correct values for the sliders
        splitMax = len(internal_nodes) - 1
        splitMarks = {str(i): str(i) for i in range(len(internal_nodes))}

        splitPMin = data_matrix["PC1"].min()
        splitPMax = data_matrix["PC1"].max()

        # create visualization points
        category_order = {
            "cluster": [
                str(i) for i in range(len(np.unique(data_matrix["cluster"])))
            ]
        }
        color_map = matplotlib.cm.get_cmap("tab20", number_of_nodes)
        colList = {str(i): util.convert_to_hex(color_map(i)) for i in range(color_map.N)}

        with open(data["new_input_object"], "rb") as obj_file:
            obj = pickle.load(obj_file)

        # Check data compatibility with the function
        if isinstance(obj, dePDDP):
            curent_figure = util.int_make_scatter_n_hist(
                data_matrix,
                splitpoint,
                obj.split_data_bandwidth_scale,
                category_order,
                colList,
            )
        else:
            curent_figure = util.int_make_scatter_n_marginal_scatter(
                data_matrix,
                splitpoint,
                category_order,
                colList
            )

    else:
        # reconstruct the already created figure as figure from dict
        curent_figure = go.Figure(
            data=curent_figure["data"],
            layout=go.Layout(curent_figure["layout"])
        )

        # ensure correct values for the sliders
        splitMax = maximum_number_splits
        splitMarks = split_marks

        splitpoint = splitpoint_position
        splitPMin = splitpoint_minimum
        splitPMax = splitpoint_max

    return (
        dcc.Graph(
            id="splitpoint_Scatter",
            figure=curent_figure,
            config={"displayModeBar": False},
        ),
        splitMax,
        splitMarks,
        splitpoint,
        splitPMin,
        splitPMax,
    )


def Splitpoint_Manipulation(object_path):
    """
    Scatter plot with splitpoint visualization creation function. This
    function is used on the initial visualization of the data and can be
    accessed throughout the execution of the interactive visualization server.

    Parameters
    ----------
    object_path : str
        The location of the temporary file containing the pickel dump of the
        object we want to visualize.

    """

    # get the necessary data for the visulization
    (
        data_matrix,
        splitpoint,
        internal_nodes,
        number_of_nodes,
    ) = util.data_preparation(
        object_path, 0
    )

    # create an ordering for the legend of the visualization
    category_order = {
        "cluster": [str(i) for i in np.unique(data_matrix["cluster"])]
    }

    # generate the colors to be used
    color_map = matplotlib.cm.get_cmap("tab20", number_of_nodes)
    colList = {
        str(i): util.convert_to_hex(color_map(i)) for i in range(color_map.N)
    }

    with open(object_path, "rb") as obj_file:
        obj = pickle.load(obj_file)

    # Check data compatibility with the function
    if isinstance(obj, dePDDP):
        figure = util.int_make_scatter_n_hist(
            data_matrix,
            splitpoint,
            obj.split_data_bandwidth_scale,
            category_order,
            colList,
        )
    else:
        figure = util.int_make_scatter_n_marginal_scatter(
            data_matrix,
            splitpoint,
            category_order,
            colList,
        )

    # Markdown description of the figure
    description = dcc.Markdown(
        util.message_center("des:splitpoitn_man", object_path),
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

    # Create the splitpoint slider
    splitSlider = html.Div(
        [
            "Select splitpoint: ",
            dcc.Slider(
                id="splitpoint_Manipulation",
                min=data_matrix["PC1"].min() - 0.001,
                max=data_matrix["PC1"].max() + 0.001,
                marks={
                    splitpoint: {
                        'label': 'Original Splitpoint',
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
            "width": "609px",
            "margin": "0px",
            "padding": "25px 0px 0px 122px",
        },
    )

    # Create the apply button for the dmanipulation of the splitpoint
    applyButton = html.Button("Apply", id="splitpoint_Man_apply", n_clicks=0)

    return html.Div([
        description,
        splitStep,
        scatter,
        splitSlider,
        applyButton,
    ])
