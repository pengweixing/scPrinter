from __future__ import annotations
from . import motifs
from .utils import regionparser
from .getFootprint import fastMultiScaleFootprints
from .getTFBS import fastRegionBindingScore
from .io import PyPrinter
from .plotting_seq import plot_a_for_plotly, plot_c_for_plotly, plot_g_for_plotly, plot_t_for_plotly
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pyBigWig
from plotly.subplots import make_subplots
import time
from dna_features_viewer import GraphicFeature, GraphicRecord
from dna_features_viewer.compute_features_levels import compute_features_levels
from scanpy.plotting.palettes import default_20, default_28, default_102
import warnings
import math
import torch

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}

def _create_motif_match_df(motif, chrom, s, e):
    # motif_matchs = motif.scan_motif([[chrom, int(s), int(e), "+"]],
    #                                 clean=False, strand=True)
    motif_matchs = motif.scan_once(chrom, int(s), int(e), clean=True, strand=True)
    if len(motif_matchs) == 0:
        return None

    features = [GraphicFeature(start=match[7],
                               end=match[8],
                               strand=match[6],
                               color=match[4],
                               label=match[4]) for match in motif_matchs]
    # record = GraphicRecord(sequence_length=init_e - init_s, features=features)
    rec = compute_features_levels(features)
    df = pd.DataFrame(
        [dict(Level=rec[key], Start=key.start + s, Finish=key.end + s, TF=str(key.label)) for key in rec])
    df.index = df['TF']
    df['delta'] = df['Finish'] - df['Start']
    return df

def _create_motif_match_trace(motif, chrom, s, e):
    tfs = list(motif.tfs)
    max_level = 0
    if len(tfs) <= 20:
        color = default_20
    elif len(tfs) <= 28:
        color = default_28
    else:
        color = (list(default_102) + list(default_20) + list(default_28) +
                 list(default_102) + list(default_20) + list(default_28))
        while len(color) < len(tfs):
            color += color
    pal = {tf:c for (tf, c) in zip(tfs, color)}

    df = _create_motif_match_df(motif, chrom, s, e)
    traces = []
    for tf in tfs[::-1]:
        try:
            tf_df = df.loc[tf]
            base, x, y = np.array(tf_df['Start']), np.array(tf_df['delta']), np.array(tf_df['Level'])
            max_level = max(max_level, np.max(y))
        except Exception as e:
            # print (e)
            base, x, y = [], [], []

        trace = go.Bar({
            'alignmentgroup': 'True',
            'base': base,
            'hovertemplate': 'TF='+tf+'<br>Start=%{base:,}<br>End=%{x:,}<extra></extra>',
            'legendgroup': tf,
            'marker': {'color': pal[tf], 'pattern': {'shape': ''}},
            'name': tf,
            'offsetgroup': tf,
            'orientation': 'h',
            'showlegend': False,
            'textposition': 'auto',
            'x': x,
            'xaxis': 'x',
            'y': y,
            'yaxis': 'y',
             'width': 1
        })

        traces.append(trace)

    return traces, max_level

def _create_gene_match_df(gffdb, chrom, s, e):
    genes = list(gffdb.region(seqid=chrom, start=s, end=e))
    if len(genes) == 0:
        return None
    feats = []
    for f in genes:
        if f.featuretype == 'gene':
            feats.append(GraphicFeature(
                start=f.start,
                end=f.end,
                strand=f.strand,
                color=f.strand,
                label=f.attributes['gene_name'][0]
            ))
    rec = compute_features_levels(feats)
    df = pd.DataFrame(
        [dict(Level=rec[key], Start=key.start, Finish=key.end, Gene=str(key.label), Strand=str(key.strand)) for key in rec])
    df['delta'] = df['Finish'] - df['Start']
    return df

def _create_gene_match_trace(gffdb, chrom, s, e, level_offset=0):
    df = _create_gene_match_df(gffdb, chrom, s, e)
    try:
        base, x, y, gene, strand = (np.array(df['Start']), np.array(df['delta']),
                                    np.array(df['Level']), np.array(df['Gene']),
                                    np.array(df['Strand']))
        y += level_offset
    except:
        base, x, y, gene, strand = [], [], [], [], []
    gene = [xx + "(" + yy + ")" for xx,yy in zip(gene, strand)]
    all_neg = all([yy == '-' for yy in strand])
    trace = go.Bar({
        'alignmentgroup': 'True',
        'base': base,
        'text': gene,
        'hovertemplate': '%{text}<br>Start=%{base:,}<br>End=%{x:,}<extra></extra>',
        'legendgroup': 'Gene',
        'marker': {'color': '#808080', 'pattern': {'shape': ''}},
        'name': 'Gene',
        'offsetgroup': 'Gene',
        'orientation': 'h',
        'showlegend': True,
        'textposition': 'inside',
        'insidetextanchor': 'end' if all_neg else 'start',
        'x': x,
        'xaxis': 'x',
        'y': y,
        'yaxis': 'y'
    })
    return trace



def plotly_plot_weights(array,
                        height_padding_factor=0.2,
                        clip_percent=0.01,
                        colors=default_colors,
                        plot_funcs={"A": plot_a_for_plotly,
                                    "C": plot_c_for_plotly,
                                    "G": plot_g_for_plotly,
                                    "T": plot_t_for_plotly},
                        highlight={},
                        start_pos=0,
                        xref='x',
                        yref='y'):
    # Initial checks on array
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if (array.shape[0] == 4 and array.shape[1] != 4):
        array = array.transpose(1, 0)
    assert array.shape[1] == 4

    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []

    all_shapes = []

    array_sum_pos = np.sum(array * (array > 0), axis=-1)
    array_sum_neg = -np.sum(array * (array < 0), axis=-1)
    sum = np.concatenate([array_sum_pos, array_sum_neg], axis=-1)
    # sum = sum[sum != 0]
    cutoff = np.max(sum) * clip_percent
    array[np.abs(array) < cutoff] = 0.0

    for i in range(array.shape[0]):
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0

        for idx, val in acgt_vals:
            letter = ["A", "C", "G", "T"][idx]
            plot_func = plot_funcs[letter]
            color = colors[idx]

            if val > 0:
                base = positive_height_so_far
                positive_height_so_far += val
            else:
                base = negative_height_so_far
                negative_height_so_far += val
            if np.abs(val) > 0:
                all_shapes.extend(plot_func(base=base, left_edge=i+start_pos, height=val, color=color,
                                            xref=xref, yref=yref))

            max_pos_height = max(max_pos_height, positive_height_so_far)
            min_neg_height = min(min_neg_height, negative_height_so_far)
            heights_at_positions.append(positive_height_so_far)
            depths_at_positions.append(negative_height_so_far)
    # print(len(all_shapes))
    # Highlight specified positions
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])

            all_shapes.append({
                'type': 'rect',
                'x0': start_pos,
                'y0': min_depth,
                'x1': end_pos,
                'y1': max_height,
                'line': {
                    'color': color,
                    'width': 2
                },
                'xref': xref,
                'yref': yref,
                'fillcolor': 'transparent'
            })

    height_padding = max(abs(min_neg_height) * height_padding_factor,
                         abs(max_pos_height) * height_padding_factor)

    return all_shapes, min_neg_height, max_pos_height, height_padding


def parse_seq_importance(seq, importance):
    v = np.zeros((len(seq), 4))
    dict = {"A":0, "C":1, "G":2, "T":3}
    for i, (s, imp) in enumerate(zip(seq, importance)):
        v[i, dict[s]] = imp
    return v


def fetch_importance_values(printer, importance, chrom, start, end):
    if type(importance) is pyBigWig.pyBigWig:
        return importance.values(chrom, start, end)
    elif type(importance) is str:
        return pyBigWig.open(importance, 'r').values(chrom, start, end)
    else:
        pad = importance.padding
        seq = printer.genome.fetch_seq(chrom, start-pad, end+pad)
        # print (seq)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            v =  importance.attribute(seq, additional_forward_args=end-start).detach().cpu().numpy()
            # print(v, pad)
            if pad != 0:
                v = v[pad:-pad]
            # print (v)
            return v

def sync_footprints(printer: PyPrinter,
                    group_names: str | list[str],
                    init_region: str,
                    motif_scanner: motifs.Motifs | None = None,
                    plotgenes: bool = False,
                    seq_importance: str | list[str] | None = None,
                    width: int = 600,
                    vmin: float = 0.5,
                    vmax: float = 2.0,
                    ):

    """
    Synced visualization of multiscale footprints
    You must run `tl.sync_footprints` first with the same `group_names` to generate the
    `group_bigwig` in `adata.uns['group_bigwig']`

    Parameters
    ----------
    printer: PyPrinter
        The printer object.
    group_names: str | list[str]
        group_names, these must be included in the `group_bigwig` in `adata.uns['group_bigwig']`
    init_region: str
        The initial region to plot, e.g. 'chr1:1000-2000'
    motif_scanner: motifs.Motifs | None
        The motif scanner object, if None, no motif will be plotted
    plotgenes: bool
        Whether to plot reference gene annotation. Default: False
    seq_importance: str
        The bigwig file of sequence importance. If None, no sequence importance will be plotted
        When provided as a list, it has to be the same length of the list of group names. If you miss certain elements,
        put None instead.
    width: int
        The width of the plot. Default: 600

    Returns
    -------

    """

    # get info for plot tss gene
    gffdb = printer.gff_db if plotgenes else None
    if motif_scanner is None:
        tfs = []
    else:
        tfs = list(motif_scanner.tfs)
    # Set global dispersion model for footprinting
    global dispModels
    dispModels = printer.dispersionModel

    # get the initial plot region
    init_region = regionparser(init_region, printer)
    chrom, init_s, init_e = init_region['Chromosome'][0], int(init_region['Start']),int(init_region['End'])

    # prep group_names & seq importance to make them both as list
    if type(group_names) not in [np.ndarray, list]:
        group_names = [group_names]
    group_names = [str(xx) for xx in group_names]

    if type(seq_importance) not in [np.ndarray, list]:
        seq_importance = [seq_importance] * len(group_names)


    assert 'group_bigwig' in printer.insertion_file.uns, "no group_bigwig detected, call pp.sync_footprints first"
    group_bigwig = printer.insertion_file.uns['group_bigwig']
    group_bigwig = {name:pyBigWig.open(group_bigwig[name], 'r') for name in group_bigwig}

    group_seq_importance = {}
    valid_seq_importance = 0

    # Content includes:
    # - Refseq
    # - TF matching results
    # - len(group_names) x [seq_importance, multiscale footprint]

    per_foot_height = 400 * width / 600
    per_importance_height = 80 * width / 600
    per_TF_height = 30 * width / 600
    annotation_first = 1 if motif_scanner is not None or gffdb is not None else 0
    # gffdb can be regarded as 1 additional TF though
    TF_track_height = per_TF_height * ((min((len(tfs) / 2 + 1), 8) + (gffdb is not None)))


    subplot_titles = ['Annotations'] * annotation_first
    row_heights = [TF_track_height] * annotation_first
    footprint_axes = {}
    seq_axes = {}

    ct = annotation_first
    for name, importance in zip(group_names, seq_importance):
        if importance is None:
            subplot_titles.append(name)
            row_heights.append(per_foot_height)
            footprint_axes[name] = '' if ct == 0 else str(ct + 1)
            ct += 1
            continue
        elif type(importance) is str and '.bw' in importance:
            group_seq_importance[name] = pyBigWig.open(importance, 'r')
        else:
            group_seq_importance[name] = importance

        subplot_titles.append(name)
        row_heights.append(per_importance_height)
        seq_axes[name] = '' if ct == 0 else str(ct + 1)
        ct += 1
        valid_seq_importance += 1

        # elif '.h5' in importance:
        #     subplot_titles.append(name)
        #     row_heights.append(per_importance_height)
        #     valid_seq_importance += 1
        #     print("importance on the fly, not implemented yet")

        subplot_titles.append(name)
        row_heights.append(per_foot_height)
        footprint_axes[name] = '' if ct == 0 else str(ct+1)
        ct += 1

    foot_track_height = per_foot_height * len(group_names)
    importance_track_height = per_importance_height * valid_seq_importance
    fig_height = foot_track_height + TF_track_height + importance_track_height

    row_heights = list(np.array(row_heights) / np.sum(row_heights))


    fig = make_subplots(rows=len(subplot_titles),
                        cols=1,
                        subplot_titles=subplot_titles,
                        row_heights=row_heights,
                        horizontal_spacing=0,
                        vertical_spacing=(50 / per_foot_height / len(group_names)))

    if motif_scanner is not None:
        figure1_traces, max_level = _create_motif_match_trace(motif_scanner, chrom, init_s, init_e)
        for traces in figure1_traces:
            fig.append_trace(traces, row=1, col=1)
        footprint_data_start = len(figure1_traces)
    else:
        footprint_data_start = 0
        max_level = 0

    if gffdb is not None:
        fig.append_trace(_create_gene_match_trace(gffdb, chrom, init_s, init_e, max_level + 1),
                         row=1, col=1)
        footprint_data_start += 1
    if group_names[0] in group_seq_importance:
        footprint_data_start += 1
    all_shapes = []
    for i, name in enumerate(group_names):
        if name in group_seq_importance:
            importance = fetch_importance_values(printer, group_seq_importance[name], chrom, init_s, init_e)
            xref = 'x' + seq_axes[name]
            yref = 'y' + seq_axes[name]
            seq = printer.genome.fetch_seq(chrom, init_s, init_e)
            seq_imp = parse_seq_importance(seq, importance)
            (all_shapes_, min_neg_height,
             max_pos_height, height_padding) = plotly_plot_weights(seq_imp,
                                                                   xref=xref,
                                                                   yref=yref,
                                                                   start_pos=init_s)
            row = seq_axes[name]
            row = 1 if row == '' else int(row)
            # Construct the final Plotly figure with all_shapes
            fig.append_trace(go.Bar(x=[],
                                    y=[]), row=row, col=1)
            fig.layout['xaxis' + seq_axes[name]]['range'] = [init_s, init_e]
            fig.layout['yaxis' + seq_axes[name]]['range'] = [min_neg_height - height_padding,
                                                                   max_pos_height + height_padding]
            fig.layout['xaxis' + seq_axes[name]]['visible'] = False
            # fig.layout['yaxis' + seq_axes[name]]['visible'] = False
            fig.layout['xaxis' + seq_axes[name]]['matches'] = 'x'
            all_shapes += all_shapes_



        v = _bigwig_footprint(group_bigwig[name],
                              group_bigwig['bias'],
                              chrom, init_s, init_e,
                              100)
        x = np.arange(init_s, init_e)
        heatmap = go.Heatmap(z=v, x=x, zmin=vmin, zmax=vmax, colorscale="Blues",
                             showscale=True if i == 0 else False,
                             hoverinfo='skip', )
        row = footprint_axes[name]
        row = 1 if row == '' else int(row)
        # Traces_add to row = 2,3,4...
        fig.add_trace(
            heatmap,
            row=row, col=1
        )
        # Force the xrange to be matched
        fig.layout['xaxis' + footprint_axes[name]]['matches'] = 'x'

    fig.update_layout(
        shapes=all_shapes,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        # paper_bgcolor='rgba(0, 0, 0, 0)',
    )


    if annotation_first > 0:
        fig.layout['xaxis']['visible'] = False
        fig.layout['yaxis']['visible'] = False

    fig.update_yaxes(fixedrange=True)
    fig.update_xaxes(autorange=False)

    # Now footprints starts with xaxis2
    fig.layout.xaxis.range = [init_s, init_e]
    fig.layout.title = f"{chrom}: {init_s}-{init_e}"
    fig.layout.width=width
    fig.layout.height=fig_height
    fig = go.FigureWidget(fig)
    fig.data[footprint_data_start].colorbar.len= 0.3 / len(group_names)
    fig.data[footprint_data_start].colorbar.y = 0.5
    fig.data[footprint_data_start].colorbar.ypad = 0.0
    fig.data[footprint_data_start].colorbar.yanchor = 'bottom'
    fig['layout']['barmode'] = 'overlay'

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)


    def change_data(self, xrange):
        xrange_zoom_min, xrange_zoom_max = fig.layout.xaxis.range[0], fig.layout.xaxis.range[1]
        s, e = int(xrange_zoom_min), int(xrange_zoom_max)
        max_level = 0
        trace = -1
        update_track = {}
        with fig.batch_update():
            if motif_scanner is not None:
                df = _create_motif_match_df(motif_scanner, chrom, s, e)
                if df is not None:
                    df.index = df['TF']
                for trace, tf in enumerate(tfs[::-1]):
                    try:
                        tf_df = df.loc[tf]
                        base, x, y = np.array(tf_df['Start']), np.array(tf_df['delta']), np.array(tf_df['Level'])
                        max_level = max(max_level, np.max(y))
                    except:
                        base, x, y = [], [], []
                    update_track[trace] = ({'x':x, 'y':y, 'base':base})
            if gffdb is not None:
                df = _create_gene_match_df(gffdb, chrom, s, e)
                try:
                    base, x, y, gene, strand = (np.array(df['Start']), np.array(df['delta']),
                                                np.array(df['Level']), np.array(df['Gene']),
                                                np.array(df['Strand']))
                    y = y + float(max_level) + 1
                except:
                    base, x, y, gene, strand = [], [], [], [], []
                gene = [xx + "(" + yy + ")" for xx, yy in zip(gene, strand)]
                update_track[trace+1] = ({
                    'x':x,
                    'y':y,
                    'base':base,
                    'text':gene})

            if (e - s) > 1000:
                step = int(math.ceil((e - s) / 1000))
            else:
                step = 1
            x = np.arange(s, e, step)
            offset = 0
            all_shapes = []
            for i, name in enumerate(group_names):
                if name in group_seq_importance:
                    importance = fetch_importance_values(printer,
                                                          group_seq_importance[name],
                                                          chrom, s, e)
                    importance = np.nan_to_num(importance)
                    seq = printer.genome.fetch_seq(chrom, s, e)
                    seq_imp = parse_seq_importance(seq, importance)
                    xref = 'x' + seq_axes[name]
                    yref = 'y' + seq_axes[name]
                    (all_shapes_, min_neg_height,
                     max_pos_height, height_padding) = plotly_plot_weights(seq_imp,
                                                                           xref=xref,
                                                                           yref=yref,
                                                                           start_pos=s)
                    fig.layout['yaxis' + seq_axes[name]]['range'] = [min_neg_height - height_padding,
                                                                           max_pos_height + height_padding]
                    if i > 0:
                        offset += 1
                    all_shapes += all_shapes_


                v = _bigwig_footprint(group_bigwig[name],
                                      group_bigwig['bias'],
                                      chrom, s, e,
                                      100)[:, ::step]

                update_track[footprint_data_start + i + offset] = ({'z':v,
                                                      'x':x})



            for trace in update_track:
                fig.data[trace].update(update_track[trace])

            fig.layout.title = f"{chrom}: {s:,}-{e:,}"
            if len(all_shapes) > 0:
                # Construct the final Plotly figure with all_shapes
                # fig.layout['xaxis%d' % (annotation_first)]['range'] = [s, e]

                fig.layout['shapes'] = all_shapes

        return fig
    # print (fig)
    fig.layout.on_change(change_data, 'xaxis.range')

    return fig


def _bigwig_bindingscore(insertion, bias, model_key, chrom, s, e, pad=0,  extra=None):
    b = np.array(bias.values(chrom, s - pad, e + pad))
    b[np.isnan(b)] = 0.0

    a = np.array(insertion.values(chrom, s - pad, e + pad))
    a[np.isnan(a)] = 0.0

    v = fastRegionBindingScore(
        a[None],
        b,
        dispModels,
        bindingScoreModels[model_key],
        contextRadius=100)[0]
    # print (v.shape, pad, a.shape)
    # print (nan1, nan2)
    # if pad > 0:
    #     v = v[pad:-pad]
    if extra is not None:
        return v, extra
    return v



def _bigwig_footprint(insertion, bias, chrom, s, e, pad=0, extra=None):
    # insertion = pyBigWig.open(insertion, 'r')
    # bias = pyBigWig.open(bias, 'r')
    b = np.array(bias.values(chrom, s - pad, e + pad))
    nan1 = np.sum(np.isnan(b))
    b[np.isnan(b)] = 0.0

    a = np.array(insertion.values(chrom, s - pad, e + pad))
    a[np.isnan(a)] = 0.0
    nan2 = np.sum(np.isnan(a))
    v = fastMultiScaleFootprints(a[None],
                                 b,
                                 dispModels,
                                 modes=np.arange(2, 101)
                                 )[0]
    # print (nan1, nan2)
    if pad > 0:
        v = v[:, pad:-pad]
    if extra is not None:
        return v, extra
    return v


def sync_footprints_advanced(printer: PyPrinter,
                    init_region: str,
                    contents: list,
                    width: int = 600,
                    vmin: float = 0.5,
                    vmax: float = 2.0,
                    ):

    """
    Synced visualization of multiscale footprints
    You must run `tl.sync_footprints` first with the same `group_names` to generate the
    `group_bigwig` in `adata.uns['group_bigwig']`

    Parameters
    ----------
    printer: PyPrinter
        The printer object.
    init_region: str
        The initial region to plot, e.g. 'chr1:1000-2000'
    contents: list
        The list of contents to plot, each element is a tuple of (type, name, arguments)
        type can be: footprint, signal, sequence, gene, motif, bindingscore, seq2signal, insertion,
    width: int
        The width of the plot. Default: 600

    Returns
    -------

    """

    # group_names: str | list[str],
    # motif_scanner: motifs.Motifs | None = None,
    # plotgenes: bool = False,
    # seq_importance: str | list[str] | None = None,

    # get info for plot tss gene

    # Set global dispersion model for footprinting
    global dispModels, bindingScoreModels
    dispModels = printer.dispersionModel
    bindingScoreModels = printer.bindingScoreModel

    # get the initial plot region
    init_region = regionparser(init_region, printer)
    chrom, init_s, init_e = init_region['Chromosome'][0], int(init_region['Start']),int(init_region['End'])

    assert 'group_bigwig' in printer.insertion_file.uns, "no group_bigwig detected, call pp.sync_footprints first"
    group_bigwig = printer.insertion_file.uns['group_bigwig']
    group_bigwig = {name:pyBigWig.open(group_bigwig[name], 'r') for name in group_bigwig}

    group_seq_importance = {}
    valid_seq_importance = 0

    # Content includes:
    # - Refseq
    # - TF matching results
    # - len(group_names) x [seq_importance, multiscale footprint]

    per_foot_height = 500 * width / 600
    per_importance_height = 75 * width / 600
    per_TF_height = 30 * width / 600
    factor = 10
    # validate_content:
    gffdb = 0
    tfs = []
    row_heights = []
    pad_heights = []
    subplot_titles = []
    specs = []


    for i, (type_,name, attr) in enumerate(contents):
        subplot_titles.append(name)
        if type_  ==  'gene':
            gffdb = printer.gff_db
            row_heights.append(per_TF_height * 2)
            pad_heights.append([factor, factor])
        elif type_ == 'motif':
            tfs = list(attr.tfs)
            row_heights.append(per_TF_height * min((len(tfs) / 2 + 1), 8))
            pad_heights.append([factor, factor * 0.5])
        elif type_ == 'sequence':
            row_heights.append(per_importance_height)
            pad_heights.append([factor, factor])
            pass
        elif type_ == 'footprint':
            row_heights.append(per_foot_height)
            pad_heights.append([factor, factor*3])
            pass
        elif type_ == 'signal':
            row_heights.append(per_importance_height)
            pad_heights.append([factor, factor])
            pass
        elif type_ == 'seq2signal':
            row_heights.append(per_importance_height)
            pad_heights.append([factor, factor])
            pass
        elif type_ == 'bindingscore':
            row_heights.append(per_importance_height)
            pad_heights.append([factor, factor])
            pass
        elif type_ == 'insertion':
            row_heights.append(per_importance_height)
            pad_heights.append([factor, factor])
        else:
            raise ValueError(f"Unknown type {type_}")

    fig_height = np.sum(row_heights) #+ np.sum(pad_heights)
    pad_heights = np.array(pad_heights)
    row_heights = list(np.array(row_heights) / np.sum(row_heights))
    for i in range(len(pad_heights)):
        specs.append([{'t':pad_heights[i][0] / fig_height, 'b':pad_heights[i][1] / fig_height}])
    # print (specs)
    fig = make_subplots(rows=len(subplot_titles),
                        cols=1,
                        subplot_titles=subplot_titles,
                        row_heights=row_heights,
                        horizontal_spacing=0,
                        vertical_spacing=0,
                        specs=specs
                        )

    all_shapes = []
    first_foot, encounter_first_foot = 0, False
    traces_start = [0]
    for i, (type_,name, attr) in enumerate(contents):
        subplot_titles.append(name)
        postfix = '' if i == 0 else str(i+1)
        if type_  ==  'gene':
            fig.append_trace(_create_gene_match_trace(gffdb, chrom, init_s, init_e),
                             row=i+1, col=1)
            fig.layout['xaxis%s' %postfix]['visible'] = False
            fig.layout['yaxis%s' %postfix]['visible'] = False
            if not encounter_first_foot:
                first_foot += 1
            traces_start.append(traces_start[-1] + 1)
            # raise NotImplementedError
        elif type_ == 'motif':
            motif_scanner = attr
            figure1_traces, max_level = _create_motif_match_trace(motif_scanner, chrom, init_s, init_e)
            for traces in figure1_traces:
                fig.append_trace(traces, row=i+1, col=1)
            if not encounter_first_foot:
                first_foot += len(figure1_traces)
            traces_start.append(traces_start[-1] + len(figure1_traces))
            fig.layout['xaxis%s' % postfix]['visible'] = False
            fig.layout['yaxis%s' % postfix]['visible'] = False
            # raise NotImplementedError
        elif type_ == 'sequence':
            importance = fetch_importance_values(printer, attr, chrom, init_s, init_e)
            xref = 'x%s' % postfix
            yref = 'y%s' % postfix
            seq = printer.genome.fetch_seq(chrom, init_s, init_e)
            seq_imp = parse_seq_importance(seq, importance)
            (all_shapes_, min_neg_height,
             max_pos_height, height_padding) = plotly_plot_weights(seq_imp,
                                                                   xref=xref,
                                                                   yref=yref,
                                                                   start_pos=init_s)
            # Construct the final Plotly figure with all_shapes
            fig.append_trace(go.Bar(x=[],
                                    y=[]), row=i+1, col=1)
            fig.layout['xaxis%s'%postfix]['range'] = [init_s, init_e]
            fig.layout['yaxis%s'%postfix]['range'] = [min_neg_height - height_padding,
                                                             max_pos_height + height_padding]
            fig.layout['xaxis%s'%postfix]['visible'] = False

            all_shapes += all_shapes_
            if not encounter_first_foot:
                first_foot += 1
            traces_start.append(traces_start[-1] + 1)
            # raise NotImplementedError
        elif type_ == 'footprint':
            v = _bigwig_footprint(group_bigwig[attr],
                                  group_bigwig['bias'],
                                  chrom, init_s, init_e,
                                  100)
            x = np.arange(init_s, init_e)
            heatmap = go.Heatmap(z=v, x=x, zmin=vmin, zmax=vmax, colorscale="Blues",
                                 showscale=True if not encounter_first_foot else False,
                                 hoverinfo='skip', )
            encounter_first_foot = True
            # Traces_add to row = 2,3,4...
            fig.add_trace(
                heatmap,
                row=i+1, col=1
            )
            traces_start.append(traces_start[-1] + 1)

            # raise NotImplementedError
        elif type_ == 'signal':
            v = pyBigWig.open(attr, 'r').values(chrom, init_s, init_e, numpy=True)
            x = np.arange(init_s, init_e)
            mask = ~np.isnan(v)
            fig.append_trace(go.Bar(x=x[mask],
                                    y=v[mask],
                                    showlegend=False, width=1,marker_line_width=0), row=i + 1, col=1)
            fig.layout['xaxis%s' % postfix]['range'] = [init_s, init_e]
            fig.layout['yaxis%s' % postfix]['range'] = [np.quantile(v[mask], 0.01), np.quantile(v[mask], 0.99)]
            fig.layout['xaxis%s' % postfix]['visible'] = False
            if not encounter_first_foot:
                first_foot += 1
            traces_start.append(traces_start[-1] + 1)

            # raise NotImplementedError
        elif type_ == 'seq2signal':
            model = attr
            with torch.no_grad():
                pad = model.padding
                seq = printer.genome.fetch_seq(chrom, init_s - pad, init_e + pad)
                v = model(seq, init_e-init_s).cpu().numpy()
                v[np.isnan(v)] = 0.0
                x = np.arange(init_s, init_e)
                fig.append_trace(go.Bar(x=x,
                                        y=v,
                                        showlegend=False, width=1,marker_line_width=0), row=i + 1, col=1)
                fig.layout['xaxis%s' % postfix]['range'] = [init_s, init_e]
                fig.layout['yaxis%s' % postfix]['range'] = [np.quantile(v, 0.01), np.quantile(v, 0.99)]
                fig.layout['xaxis%s' % postfix]['visible'] = False
                if not encounter_first_foot:
                    first_foot += 1
                traces_start.append(traces_start[-1] + 1)

        elif type_ == 'bindingscore':
            group_name, model_key = attr
            v = _bigwig_bindingscore(group_bigwig[group_name],
                                    group_bigwig['bias'],
                                    model_key,
                                    chrom, init_s, init_e,
                                    100)
            x = np.arange(init_s, init_e)
            v[np.isnan(v)] = 0.0
            fig.append_trace(go.Bar(x=x,
                                    y=v,
                                    showlegend=False, width=1,marker_line_width=0), row=i + 1, col=1)
            fig.layout['xaxis%s' % postfix]['range'] = [init_s, init_e]
            fig.layout['yaxis%s' % postfix]['range'] = [np.quantile(v, 0.01), np.quantile(v, 0.99)]
            fig.layout['xaxis%s' % postfix]['visible'] = False
            if not encounter_first_foot:
                first_foot += 1
            traces_start.append(traces_start[-1] + 1)
        elif type_ == 'insertion':
            v = group_bigwig[attr].values(chrom, init_s, init_e, numpy=True)
            x = np.arange(init_s, init_e)
            mask = ~np.isnan(v)
            fig.append_trace(go.Bar(x=x[mask],
                                    y=v[mask],
                                    showlegend=False, width=1, marker_line_width=0), row=i + 1, col=1)
            fig.layout['xaxis%s' % postfix]['range'] = [init_s, init_e]
            fig.layout['yaxis%s' % postfix]['range'] = [np.quantile(v[mask], 0.01), np.quantile(v[mask], 0.99)]
            fig.layout['xaxis%s' % postfix]['visible'] = False
            if not encounter_first_foot:
                first_foot += 1
            traces_start.append(traces_start[-1] + 1)
        else:
            raise ValueError(f"Unknown type {type_}")
        if i > 0:
            # Force the xrange to be matched
            fig.layout['xaxis%s' % postfix]['matches'] = 'x'
            fig.layout['xaxis%s' % postfix]['tickformat'] = ',d'
            fig.layout['xaxis%s' % postfix]['ticks'] = 'inside'

    traces_start = traces_start[:-1]
    fig.update_layout(
        shapes=all_shapes,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        # paper_bgcolor='rgba(0, 0, 0, 0)',
    )


    # if annotation_first > 0:
        # fig.layout['xaxis']['visible'] = False
        # fig.layout['yaxis']['visible'] = False

    fig.update_yaxes(fixedrange=True)
    fig.update_xaxes(autorange=False)

    # Now footprints starts with xaxis2
    fig.layout.xaxis.range = [init_s, init_e]
    fig.layout.title = f"{chrom}: {init_s}-{init_e}"
    fig.layout.width=width
    fig.layout.height=fig_height
    fig = go.FigureWidget(fig)

    if encounter_first_foot:
        fig.data[first_foot].colorbar.len = 0.1 #/ len(contents)
        fig.data[first_foot].colorbar.y = 0.5
        fig.data[first_foot].colorbar.ypad = 0.0
        fig.data[first_foot].colorbar.yanchor = 'bottom'
    fig['layout']['barmode'] = 'overlay'

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)


    def change_data(self, xrange):
        xrange_zoom_min, xrange_zoom_max = fig.layout.xaxis.range[0], fig.layout.xaxis.range[1]
        s, e = int(xrange_zoom_min), int(xrange_zoom_max)
        max_level = 0
        update_track = {}
        with fig.batch_update():
            if (e - s) > 1000:
                step = int(math.ceil((e - s) / 1000))
            else:
                step = 1

            all_shapes = []
            for i, (type_, name, attr) in enumerate(contents):
                subplot_titles.append(name)
                postfix = '' if i == 0 else str(i + 1)
                if type_ == 'gene':
                    df = _create_gene_match_df(gffdb, chrom, s, e)
                    try:
                        base, x, y, gene, strand = (np.array(df['Start']), np.array(df['delta']),
                                                    np.array(df['Level']), np.array(df['Gene']),
                                                    np.array(df['Strand']))
                        y = y
                    except:
                        base, x, y, gene, strand = [], [], [], [], []
                    gene = [xx + "(" + yy + ")" for xx, yy in zip(gene, strand)]
                    update_track[traces_start[i]] = ({
                        'x': x,
                        'y': y,
                        'base': base,
                        'text': gene})
                elif type_ == 'motif':
                    motif_scanner = attr
                    df = _create_motif_match_df(motif_scanner, chrom, s, e)
                    if df is not None:
                        df.index = df['TF']
                    for trace, tf in enumerate(tfs[::-1]):
                        try:
                            tf_df = df.loc[tf]
                            base, x, y = np.array(tf_df['Start']), np.array(tf_df['delta']), np.array(tf_df['Level'])
                            # max_level = max(max_level, np.max(y))
                        except:
                            base, x, y = [], [], []
                        update_track[trace+traces_start[i]] = ({'x': x, 'y': y, 'base': base})
                    # raise NotImplementedError
                elif type_ == 'sequence':
                    importance = fetch_importance_values(printer, attr, chrom, s, e)
                    importance = np.nan_to_num(importance)
                    seq = printer.genome.fetch_seq(chrom, s, e)
                    seq_imp = parse_seq_importance(seq, importance)
                    xref = 'x' + postfix
                    yref = 'y' + postfix
                    (all_shapes_, min_neg_height,
                     max_pos_height, height_padding) = plotly_plot_weights(seq_imp,
                                                                           xref=xref,
                                                                           yref=yref,
                                                                           start_pos=s)
                    fig.layout['yaxis' + postfix]['range'] = [min_neg_height - height_padding,
                                                                     max_pos_height + height_padding]
                    all_shapes += all_shapes_
                    update_track[traces_start[i]] = ({})
                    # raise NotImplementedError
                elif type_ == 'footprint':
                    x = np.arange(s, e, step)
                    v = _bigwig_footprint(group_bigwig[attr],
                                          group_bigwig['bias'],
                                          chrom, s, e,
                                          100)[:, ::step]
                    update_track[traces_start[i]] = ({'z': v, 'x': x})
                elif type_ == 'signal':
                    v = pyBigWig.open(attr, 'r').values(chrom, s, e, numpy=True)
                    mask = ~np.isnan(v)
                    x = np.arange(s, e)[mask]
                    v = v[mask]
                    update_track[traces_start[i]] = ({'x':x, 'y':v})
                    fig.layout['yaxis%s' % postfix]['range'] = [np.quantile(v, 0.01),
                                                                np.quantile(v, 0.99)]
                elif type_ == 'seq2signal':
                    model = attr
                    with torch.no_grad():
                        pad = model.padding
                        seq = printer.genome.fetch_seq(chrom, s - pad, e + pad)
                        v = model(seq, e-s).cpu().numpy()
                        # print (v.shape, pad, s, e)
                        update_track[traces_start[i]] = ({'x': np.arange(s, e), 'y': v})
                        fig.layout['yaxis%s' % postfix]['range'] = [np.quantile(v, 0.01),
                                                                    np.quantile(v, 0.99)]

                elif type_ == 'bindingscore':
                    group_name, model_key = attr
                    v = _bigwig_bindingscore(group_bigwig[group_name],
                                             group_bigwig['bias'],
                                             model_key,
                                             chrom, s, e,
                                             100)
                    v[np.isnan(v)] = 0.0
                    update_track[traces_start[i]] = ({'x':np.arange(s,e), 'y':v})
                    fig.layout['yaxis%s' % postfix]['range'] = [np.quantile(v, 0.01),
                                                                np.quantile(v, 0.99)]
                elif type_ == 'insertion':
                    v = group_bigwig[attr].values(chrom, s, e, numpy=True)
                    mask = ~np.isnan(v)
                    x = np.arange(s, e)[mask]
                    v = v[mask]
                    update_track[traces_start[i]] = ({'x': x, 'y': v})
                    fig.layout['yaxis%s' % postfix]['range'] = [np.quantile(v, 0.01),
                                                                np.quantile(v, 0.99)]
                else:
                    raise ValueError(f"Unknown type {type_}")



            for trace in update_track:
                fig.data[trace].update(update_track[trace])

            fig.layout.title = f"{chrom}: {s:,}-{e:,}"
            if len(all_shapes) > 0:
                # Construct the final Plotly figure with all_shapes
                # fig.layout['xaxis%d' % (annotation_first)]['range'] = [s, e]

                fig.layout['shapes'] = all_shapes

        return fig
    # print (fig)
    fig.layout.on_change(change_data, 'xaxis.range')

    return fig
