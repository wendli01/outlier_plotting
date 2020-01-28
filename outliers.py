"""
More graceful handling of outliers for seaborn plotting.
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn.categorical import _CategoricalPlotter
import pandas as pd


def _plot_outliers(ax, outliers, plot_extents, orient='v', group=0, padding=.05, margin=.1, outlier_hues=None,
                   fmt='.2g'):
    def _vals_to_str(vals, val_categories):
        def _format_val(val):
            return ("{:" + fmt + "}").format(val)

        if val_categories is None:
            vals = sorted(vals, reverse=True)
            return '\n'.join([_format_val(val) for val in vals])

        texts = []
        df = pd.DataFrame({'val': vals, 'cat': val_categories})
        for cat in sorted(df.cat.unique()):
            cat_vals = df[df.cat == cat].val
            texts.append(str(cat) + ':\t' + '\n\t'.join([_format_val(val) for val in cat_vals]))
        return '\n'.join(texts).expandtabs()

    def _add_margin(lo, hi, mrg=.1, rng=None):
        rng = hi - lo if rng is None else rng
        return lo - mrg * rng, hi + mrg * rng

    def _set_limits(t):
        def _get_bbox():
            plt.gcf().canvas.draw()
            return t.get_window_extent().inverse_transformed(plt.gca().transData)

        old_extents = ax.get_ylim() if is_v else ax.get_xlim()

        val_coords = np.array(_get_bbox()).T[dim_sel]
        val_coords = _add_margin(*val_coords, margin, rng=np.diff(plot_extents[dim_sel]))

        new_extents = [np.min([val_coords, old_extents]), np.max([val_coords, old_extents])]
        lim_setter = ax.set_ylim if is_v else ax.set_xlim
        lim_setter(new_extents)

    def _plot_text(is_low):
        is_relevant = outliers < vmin if is_low else outliers > vmax
        points = outliers[is_relevant]
        point_hues = None if outlier_hues is None else outlier_hues[is_relevant]
        if not len(points) > 0:
            return

        val_pos = _add_margin(vmin, vmax, padding)[0 if is_low else 1]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.1)
        text = _vals_to_str(points, point_hues)

        if is_v:
            t = ax.text(group, val_pos, text, ha='center', multialignment='right',
                        va='top' if is_low else 'bottom', bbox=props)

        else:
            t = ax.text(val_pos, group, text, va='center', multialignment='right',
                        ha='right' if is_low else 'left', bbox=props)

        _set_limits(t)

    is_v = orient == 'v'
    dim_sel = 1 if is_v else 0
    vmin, vmax = plot_extents[dim_sel]
    _plot_text(True), _plot_text(False)


def handle_outliers(data: pd.DataFrame, x: str = None, y: str = None, hue: str = None,
                    plotter: callable = sns.swarmplot, inlier_range: float = 1.5, padding: float = .05,
                    margin: float = .1, fmt='.2g', **kwargs) -> plt.axes:
    """
    Remove outliers from the plot and show them as text boxes. Works well with `sns.violinplot`, `sns.swarmplot`,
    `sns.boxplot` and the like. Does *not* work with axis grids.

    data: pd.DataFrame
        Dataset for plotting. Expected to be long-form.
    x: str
        names of  x variables in data
    y: str
        names of y variables in data
    hue: str
        names of hue variables in data. Not fully supported.
    plotter: callable
        `seaborn` plotting function that works with long-form data.
    whis: float
        Proportion of the IQR past the low and high quartiles to extend the original plot. Points outside this range
        will be identified as outliers.
    padding: float
        Padding in % of figure size between original plot and text boxes.
    margin: float
        Margin in % of figure size between text boxes and axis extent.
    fmt: str
        String formatting code to use when adding annotations.
    kwargs: key, value mappings
        Other keyword arguments are passed through to the plotter.
    Returns
    -------
    ax: matplotlib Axes
        The Axes object with the plot drawn onto it.

    """

    def _get_info():
        cp = _CategoricalPlotter()
        cp.establish_variables(x=x, y=y, data=data, hue=hue)
        return cp.value_label, cp.group_label, cp.group_names, cp.orient, cp.hue_names, cp.plot_hues

    def _get_cutoffs(a: np.array, quantiles=(.25, .75)):
        quartiles = np.quantile(a, list(quantiles))
        iqr = np.diff(quartiles)
        return quartiles[0] - inlier_range * iqr, quartiles[1] + inlier_range * iqr

    def _add_to_kwargs(k, v):
        if v is not None:
            kwargs[k] = v

    def _plot_group_outliers(group_idx=0, group_name=None):
        if group_name is None:
            group_values = data[value_label].values
        else:
            group_values = data[data[group_label] == group_name][value_label].values

        is_outlier = np.logical_or(cutoff_lo > group_values, group_values > cutoff_hi)
        group_outliers = group_values[is_outlier]
        outlier_hues = plot_hues[group_idx][is_outlier] if plot_hues is not None else None
        _plot_outliers(ax, group_outliers, orient=orient, group=group_idx, padding=padding, margin=margin,
                       plot_extents=plot_extents, outlier_hues=outlier_hues, fmt=fmt)

    value_label, group_label, group_names, orient, hue_names, plot_hues = _get_info()
    plot_data = data[value_label].values if value_label is not None else np.array(data)

    cutoff_lo, cutoff_hi = _get_cutoffs(plot_data)
    inlier_data = data.loc[np.logical_and(cutoff_lo <= plot_data, plot_data <= cutoff_hi)]

    if len(inlier_data) == 0:
        raise UserWarning('No inliers in data, pleas modify inlier_range!')

    _add_to_kwargs('x', x), _add_to_kwargs('y', y), _add_to_kwargs('hue', hue)

    ax = plotter(data=inlier_data, **kwargs)
    plot_extents = np.array([ax.get_xlim(), ax.get_ylim()])

    if plotter == sns.kdeplot:
        orient = 'h'
        group = .5 * np.diff(ax.get_ylim())
        outlier_data = plot_data[np.logical_or(cutoff_lo > plot_data, plot_data > cutoff_hi)]
        _plot_outliers(ax, outlier_data, orient=orient, group=group, padding=padding, margin=margin,
                       plot_extents=plot_extents, fmt=fmt)
        return ax

    if not group_names:
        _plot_group_outliers()

    for group_idx, group_name in enumerate(group_names):
        _plot_group_outliers(group_idx, group_name)

    return ax
