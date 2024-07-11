import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_a_for_plotly(base, left_edge, height, color, xref="x", yref="y", thickness=0.1):
    """
    This function generates a list of dictionaries representing the shapes of the letter 'A'
    to be plotted using Plotly.

    Parameters:
    base (float): The y-coordinate of the bottom-left corner of the letter.
    left_edge (float): The x-coordinate of the bottom-left corner of the letter.
    height (float): The height of the letter.
    color (str): The color of the letter.
    xref (str): The reference system for the x-coordinates. Default is 'x'.
    yref (str): The reference system for the y-coordinates. Default is 'y'.
    thickness (float): The thickness of the letter. Default is 0.1.

    Returns:
    list: A list of dictionaries representing the shapes of the letter 'A' to be plotted using Plotly.
    """
    shapes = []
    a_polygon_coords = [
        np.array(
            [
                [0.0, 0.0],
                [0.5, 1.0],
                [0.5, 1 - thickness],
                [thickness, 0.0],
            ]
        ),
        np.array(
            [
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, 1 - thickness],
                [1 - thickness, 0.0],
            ]
        ),
        np.array(
            [
                [0.225, 0.4 + thickness / 2],
                [0.775, 0.4 + thickness / 2],
                [0.85, 0.4 - thickness / 2],
                [0.15, 0.4 - thickness / 2],
            ]
        ),
    ]
    for polygon_coords in a_polygon_coords:
        shape_coords = (
            np.array([1, height])[None, :] * polygon_coords + np.array([left_edge, base])[None, :]
        )
        path_str = (
            "M " + " L ".join(["{},{}".format(coord[0], coord[1]) for coord in shape_coords]) + " Z"
        )
        shapes.append(
            {
                "type": "path",
                "xref": xref,
                "yref": yref,
                "path": path_str,
                "fillcolor": color,
                "line": {"color": color},
            }
        )
    return shapes


def plot_c_for_plotly(base, left_edge, height, color, xref="x", yref="y", thickness=0.1):
    shapes = []
    # The outer ellipse
    shapes.append(
        {
            "type": "circle",
            "xref": xref,
            "yref": yref,
            "x0": left_edge,
            "y0": base,
            "x1": left_edge + 1.3,
            "y1": base + height,
            "fillcolor": color,
            "line": {"color": color},
        }
    )
    # The inner ellipse
    shapes.append(
        {
            "type": "circle",
            "xref": xref,
            "yref": yref,
            "x0": left_edge + thickness * 2,
            "y0": base + 0.15 * height,
            "x1": left_edge + 1.0,
            "y1": base + 0.85 * height,
            "fillcolor": "white",
            "line": {"color": "white"},
        }
    )
    # The rectangle to cut off part of the ellipse
    shapes.append(
        {
            "type": "rect",
            "xref": xref,
            "yref": yref,
            "x0": left_edge + 1,
            "y0": base,
            "x1": left_edge + 2,
            "y1": base + height,
            "fillcolor": "white",
            "line": {"color": "white"},
        }
    )
    return shapes


def plot_g_for_plotly(base, left_edge, height, color, xref="x", yref="y", thickness=0.1):
    shapes = plot_c_for_plotly(
        base, left_edge, height, color, xref=xref, yref=yref, thickness=thickness
    )  # Start with the shapes for 'C'

    # Additional rectangle in 'G'
    shapes.append(
        {
            "type": "rect",
            "xref": xref,
            "yref": yref,
            "x0": left_edge + 0.825,
            "y0": base + 0.085 * height,
            "x1": left_edge + 0.825 + thickness / 2,
            "y1": base + 0.5 * height,
            "fillcolor": color,
            "line": {"color": color},
        }
    )

    # Another rectangle in 'G'
    shapes.append(
        {
            "type": "rect",
            "xref": xref,
            "yref": yref,
            "x0": left_edge + 0.625,
            "y0": base + (0.5 - thickness / 2) * height,
            "x1": left_edge + 0.625 + 0.2,
            "y1": base + (0.5 - thickness / 2) * height + (thickness / 2) * height,
            "fillcolor": color,
            "line": {"color": color},
        }
    )

    return shapes


def plot_t_for_plotly(base, left_edge, height, color, xref="x", yref="y", thickness=0.1):
    shapes = []

    # Vertical rectangle in 'T'
    shapes.append(
        {
            "type": "rect",
            "xref": xref,
            "yref": yref,
            "x0": left_edge + 0.45,
            "y0": base,
            "x1": left_edge + 0.45 + thickness,
            "y1": base + height,
            "fillcolor": color,
            "line": {"color": color},
        }
    )

    # Horizontal rectangle in 'T'
    shapes.append(
        {
            "type": "rect",
            "xref": xref,
            "yref": yref,
            "x0": left_edge,
            "y0": base + (1 - thickness) * height,
            "x1": left_edge + 1.0,
            "y1": base + height,
            "fillcolor": color,
            "line": {"color": color},
        }
    )

    return shapes


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array(
            [
                [0.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.8],
                [0.2, 0.0],
            ]
        ),
        np.array(
            [
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.8],
                [0.8, 0.0],
            ]
        ),
        np.array(
            [
                [0.225, 0.45],
                [0.775, 0.45],
                [0.85, 0.3],
                [0.15, 0.3],
            ]
        ),
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(
            matplotlib.patches.Polygon(
                (
                    np.array([1, height])[None, :] * polygon_coords
                    + np.array([left_edge, base])[None, :]
                ),
                facecolor=color,
                edgecolor=color,
            )
        )


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65, base + 0.5 * height],
            width=1.3,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
    )
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65, base + 0.5 * height],
            width=0.7 * 1.3,
            height=0.7 * height,
            facecolor="white",
            edgecolor="white",
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 1, base],
            width=1.0,
            height=height,
            facecolor="white",
            edgecolor="white",
            fill=True,
        )
    )


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65, base + 0.5 * height],
            width=1.3,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
    )
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65, base + 0.5 * height],
            width=0.7 * 1.3,
            height=0.7 * height,
            facecolor="white",
            edgecolor="white",
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 1, base],
            width=1.0,
            height=height,
            facecolor="white",
            edgecolor="white",
            fill=True,
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 0.825, base + 0.085 * height],
            width=0.174,
            height=0.415 * height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 0.625, base + 0.35 * height],
            width=0.374,
            height=0.15 * height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 0.4, base],
            width=0.2,
            height=height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge, base + 0.8 * height],
            width=1.0,
            height=0.2 * height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )


default_colors = {0: "green", 1: "blue", 2: "orange", 3: "red"}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}

"""
Code from chrombpnet
"""


def plot_weights_given_ax(
    ax,
    array,
    height_padding_factor=0.2,
    length_padding=1.0,
    subticks_frequency=1.0,
    highlight={},
    colors=default_colors,
    plot_funcs=default_plot_funcs,
):
    """
    Plots weights given an axis and an array of weights.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot the weights.
    array (numpy.ndarray): The array of weights to be plotted. It should be a 2D NumPy array with shape (n_positions, 4), where n_positions is the number of positions and 4 represents the weights for A, C, G, and T.
    height_padding_factor (float, optional): The factor by which to pad the height of the plot. Default is 0.2.
    length_padding (float, optional): The amount by which to pad the length of the plot. Default is 1.0.
    subticks_frequency (float, optional): The frequency at which to place subticks on the x-axis. Default is 1.0.
    highlight (dict, optional): A dictionary specifying the positions to highlight. The keys of the dictionary are the colors, and the values are lists of tuples, where each tuple represents a range of positions to highlight.
    colors (dict, optional): A dictionary specifying the colors to use for each base. Default is `default_colors`.
    plot_funcs (dict, optional): A dictionary specifying the functions to use for plotting each base. Default is `default_plot_funcs`.

    Returns:
    None
    """
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if array.shape[0] == 4 and array.shape[1] != 4:
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if letter[1] > 0:
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    xy=[start_pos, min_depth],
                    width=end_pos - start_pos,
                    height=max_height - min_depth,
                    edgecolor=color,
                    fill=False,
                )
            )

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(
        abs(min_neg_height) * (height_padding_factor),
        abs(max_pos_height) * (height_padding_factor),
    )
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)


def plot_weights(
    array,
    figsize=(20, 2),
    height_padding_factor=0.2,
    length_padding=1.0,
    subticks_frequency=1.0,
    colors=default_colors,
    plot_funcs=default_plot_funcs,
    highlight={},
):
    """
    A wrapper function for plot_weights_given_ax,
    providing a simple interface for plotting weights without creating the axes

    Parameters:
    array (numpy.ndarray): The array of weights to be plotted. It should be a 2D NumPy array with shape (n_positions, 4), where n_positions is the number of positions and 4 represents the weights for A, C, G, and T.
    figsize (tuple, optional): The size of the figure. Default is (20)
    height_padding_factor (float, optional): The factor by which to pad the height of the plot. Default is 0.2.
    length_padding (float, optional): The amount by which to pad the length of the plot. Default is 1.0.
    subticks_frequency (float, optional): The frequency at which to place subticks on the x-axis. Default is 1.0.
    highlight (dict, optional): A dictionary specifying the positions to highlight. The keys of the dictionary are the colors, and the values are lists of tuples, where each tuple represents a range of positions to highlight.
    colors (dict, optional): A dictionary specifying the colors to use for each base. Default is `default_colors`.
    plot_funcs (dict, optional): A dictionary specifying the functions to use for plotting each base. Default is `default_plot_funcs`.


    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(0, 0, 1, 1)
    plot_weights_given_ax(
        ax=ax,
        array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight,
    )
    ax.axis("off")
    plt.tight_layout()
    # return ax
    plt.show()
    return
