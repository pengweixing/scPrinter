import plotly.graph_objs as go
import pyBigWig
from plotly.subplots import make_subplots
import numpy as np

def plot_a_for_plotly(base, left_edge, height, color, xref='x', yref='y',
                      thickness=0.1):
    shapes = []
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 1-thickness],
            [thickness, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 1-thickness],
            [1-thickness, 0.0],
        ]),
        np.array([
            [0.225, 0.4+thickness/2],
            [0.775, 0.4+thickness/2],
            [0.85, 0.4-thickness/2],
            [0.15, 0.4-thickness/2]])]
    for polygon_coords in a_polygon_coords:
        shape_coords = (np.array([1, height])[None, :] * polygon_coords
                        + np.array([left_edge, base])[None, :])
        path_str = 'M ' + ' L '.join(['{},{}'.format(coord[0], coord[1]) for coord in shape_coords]) + ' Z'
        shapes.append({
            'type': 'path',
            'xref': xref,
            'yref': yref,
            'path': path_str,
            'fillcolor': color,
            'line': {
                'color': color
            }
        })
    return shapes

def plot_a_for_plotly_thick(base, left_edge, height, color, xref='x', yref='y'):
    shapes = []
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        shape_coords = (np.array([1, height])[None, :] * polygon_coords
                        + np.array([left_edge, base])[None, :])
        path_str = 'M ' + ' L '.join(['{},{}'.format(coord[0], coord[1]) for coord in shape_coords]) + ' Z'
        shapes.append({
            'type': 'path',
            'xref': xref,
            'yref': yref,
            'path': path_str,
            'fillcolor': color,
            'line': {
                'color': color
            }
        })
    return shapes


def plot_c_for_plotly(base, left_edge, height, color, xref='x', yref='y', thickness=0.1):
    shapes = []
    # The outer ellipse
    shapes.append({
        'type': 'circle',
        'xref': xref,
        'yref': yref,
        'x0': left_edge,
        'y0': base,
        'x1': left_edge + 1.3,
        'y1': base + height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })
    # The inner ellipse
    shapes.append({
        'type': 'circle',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + thickness * 2,
        'y0': base + 0.15 * height,
        'x1': left_edge + 1.0,
        'y1': base + 0.85 * height,
        'fillcolor': 'white',
        'line': {
            'color': 'white'
        }
    })
    # The rectangle to cut off part of the ellipse
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + 1,
        'y0': base,
        'x1': left_edge + 2,
        'y1': base + height,
        'fillcolor': 'white',
        'line': {
            'color': 'white'
        }
    })
    return shapes

def plot_c_for_plotly_thick(base, left_edge, height, color, xref='x', yref='y'):
    shapes = []
    # The outer ellipse
    shapes.append({
        'type': 'circle',
        'xref': xref,
        'yref': yref,
        'x0': left_edge,
        'y0': base,
        'x1': left_edge + 1.3,
        'y1': base + height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })
    # The inner ellipse
    shapes.append({
        'type': 'circle',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + 0.3,
        'y0': base + 0.15 * height,
        'x1': left_edge + 1.0,
        'y1': base + 0.85 * height,
        'fillcolor': 'white',
        'line': {
            'color': 'white'
        }
    })
    # The rectangle to cut off part of the ellipse
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + 1,
        'y0': base,
        'x1': left_edge + 2,
        'y1': base + height,
        'fillcolor': 'white',
        'line': {
            'color': 'white'
        }
    })
    return shapes


def plot_g_for_plotly(base, left_edge, height, color, xref='x', yref='y', thickness=0.1):
    shapes = plot_c_for_plotly(base, left_edge, height, color, xref=xref, yref=yref, thickness=thickness)  # Start with the shapes for 'C'

    # Additional rectangle in 'G'
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + 0.825,
        'y0': base + 0.085 * height,
        'x1': left_edge + 0.825 + thickness/2,
        'y1': base + 0.5 * height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })

    # Another rectangle in 'G'
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + 0.625,
        'y0': base + (0.5-thickness/2) * height,
        'x1': left_edge + 0.625 + 0.2,
        'y1': base + (0.5-thickness/2) * height + (thickness / 2) * height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })

    return shapes

def plot_g_for_plotly_thick(base, left_edge, height, color, xref='x', yref='y'):
    shapes = plot_c_for_plotly(base, left_edge, height, color, xref=xref, yref=yref)  # Start with the shapes for 'C'

    # Additional rectangle in 'G'
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + 0.825,
        'y0': base + 0.085 * height,
        'x1': left_edge + 0.825 + 0.174,
        'y1': base + 0.085 * height + 0.415 * height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })

    # Another rectangle in 'G'
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + 0.625,
        'y0': base + 0.35 * height,
        'x1': left_edge + 0.625 + 0.374,
        'y1': base + 0.35 * height + 0.15 * height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })

    return shapes

def plot_t_for_plotly(base, left_edge, height, color, xref='x', yref='y', thickness=0.1):
    shapes = []

    # Vertical rectangle in 'T'
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + 0.45,
        'y0': base,
        'x1': left_edge + 0.45 + thickness,
        'y1': base + height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })

    # Horizontal rectangle in 'T'
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge,
        'y0': base + (1-thickness) * height,
        'x1': left_edge + 1.0,
        'y1': base + height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })

    return shapes

def plot_t_for_plotly_thick(base, left_edge, height, color, xref='x', yref='y'):
    shapes = []

    # Vertical rectangle in 'T'
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge + 0.4,
        'y0': base,
        'x1': left_edge + 0.4 + 0.2,
        'y1': base + height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })

    # Horizontal rectangle in 'T'
    shapes.append({
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': left_edge,
        'y0': base + 0.8 * height,
        'x1': left_edge + 1.0,
        'y1': base + 0.8 * height + 0.2 * height,
        'fillcolor': color,
        'line': {
            'color': color
        }
    })

    return shapes
