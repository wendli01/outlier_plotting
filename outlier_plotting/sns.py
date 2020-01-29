"""
Outlier outlier_plotting in seaborn.
"""
import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn.categorical import _CategoricalPlotter
import pandas as pd
from typing import Union


def _plot_outliers(ax, outliers, plot_extents, orient='v', group=0, padding=.05, outlier_hues=None,
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
        def _get_bbox_pos():
            plt.gcf().canvas.draw()
            return t.get_window_extent().inverse_transformed(plt.gca().transData)

        old_extents = ax.get_ylim() if is_v else ax.get_xlim()
        val_coords = np.array(_get_bbox_pos()).T[dim_sel]

        new_extents = [np.min([val_coords, old_extents]), np.max([val_coords, old_extents])]
        lim_setter = ax.set_ylim if is_v else ax.set_xlim

        # if new extents are more than .5 times larger as old extents with padding, we assume _get_bbox_pos failed.
        if (np.diff(new_extents) / np.diff(old_extents)) > 1 + padding + .5:
            warnings.warn('Determining text position failed, cannot set new plot extent. '
                          'Please modify margin if any text is cut off.'.format(padding))
            new_extents = old_extents
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


def _add_margins(ax: plt.Axes, plot_data, cutoff_lo: float, cutoff_hi: float, orient: str, margin: float):
    old_extents = ax.get_ylim() if orient == 'v' else ax.get_xlim()
    lim_setter = ax.set_ylim if orient == 'v' else ax.set_xlim
    if np.min(plot_data) < cutoff_lo:
        lim_setter([old_extents[0] - margin * np.diff(old_extents), None])
    if np.max(plot_data) > cutoff_hi:
        lim_setter([None, old_extents[1] + margin * np.diff(old_extents)])


def _get_inlier_data(data: Union[pd.Series, pd.DataFrame, np.ndarray], plot_data, cutoff_lo: float, cutoff_hi: float) -> \
        Union[pd.Series, pd.DataFrame]:
    inlier_data = data[np.logical_and(cutoff_lo <= plot_data, plot_data <= cutoff_hi)]
    if type(inlier_data) != pd.DataFrame:
        inlier_data = pd.Series(inlier_data)
    if type(inlier_data) == pd.Series:
        inlier_data = inlier_data.reset_index(drop=True)

    if len(inlier_data) == 0:
        raise UserWarning('No inliers in data, pleas modify inlier_range!')

    return inlier_data


def handle_outliers(data: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
                    x: Union[pd.Series, np.ndarray, str] = None, y: Union[pd.Series, np.ndarray, str] = None,
                    hue: str = None, plotter: callable = sns.swarmplot, inlier_range: float = 1.5, padding: float = .05,
                    margin: float = .1, fmt='.2g', **kwargs) -> plt.axes:
    """
    Remove outliers from the plot and show them as text boxes. Works well with `sns.violinplot`, `sns.swarmplot`,
    `sns.boxplot` and the like. Does *not* work with axis grids.

    data: pd.DataFrame
        Dataset for outlier_plotting. Expected to be long-form.
    x: str
        names of  x variables in data
    y: str
        names of y variables in data
    hue: str
        names of hue variables in data. Not fully supported.
    plotter: callable
        `seaborn` outlier_plotting function that works with long-form data.
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
        cp: _CategoricalPlotter = _CategoricalPlotter()
        cp.establish_variables(x=x, y=y, data=data, hue=hue)
        orient = 'h' if plotter == sns.kdeplot else cp.orient
        return cp.value_label, cp.group_label, cp.group_names, orient, cp.hue_title, cp.plot_data

    def _get_plot_data() -> np.ndarray:
        if data is not None:
            return data[value_label].values if value_label is not None else np.array(data)
        ret = kwargs[_which_data_var()]
        return ret.values if type(ret) == pd.Series else ret

    def _which_data_var():
        if data is not None:
            return 'data'
        else:
            if x is not None and y is not None:
                return 'y' if orient == 'v' else 'x'
            return 'y' if x is None else 'x'

    def _get_cutoffs(a: np.array, quantiles=(.25, .75)):
        quartiles = np.quantile(a, list(quantiles))
        iqr = np.diff(quartiles)
        return quartiles[0] - inlier_range * iqr, quartiles[1] + inlier_range * iqr

    def _add_to_kwargs(k, v):
        if v is not None:
            kwargs[k] = v

    def _plot_group_outliers(data: Union[pd.DataFrame, pd.Series], plot_extents, ax: plt.Axes, group_idx: int = 0,
                             group_name: str = None):
        if group_name is None or group_label is None or type(data) == pd.Series:
            group_data = data
        else:
            group_data = data[data[group_label] == group_name]

        if value_label is None or type(group_data) == pd.Series:
            group_values = group_data.values
        else:
            group_values = group_data[value_label].values

        is_outlier = np.logical_or(cutoff_lo > group_values, group_values > cutoff_hi)
        group_outliers = group_values[is_outlier]
        outlier_hues = None if hue_label is None else group_data[is_outlier][hue_label]

        if all(is_outlier):
            raise UserWarning('No inliers in group <{}>, please modify inlier_range!'.format(group_name))

        _plot_outliers(ax, group_outliers, orient=orient, group=group_idx, padding=padding,
                       plot_extents=plot_extents, outlier_hues=outlier_hues, fmt=fmt)

    def _plot_ax_outliers(ax: plt.Axes, ax_data: Union[pd.DataFrame, pd.Series], plot_extents):
        if plotter == sns.kdeplot:
            group = .5 * np.diff(ax.get_ylim())
            ax_data = ax_data.values

            outlier_data = ax_data[np.logical_or(cutoff_lo > ax_data, ax_data > cutoff_hi)]
            _plot_outliers(ax, outlier_data, orient=orient, group=group, padding=padding,
                           plot_extents=plot_extents, fmt=fmt)
            return ax

        if not group_names:
            _plot_group_outliers(ax_data, plot_extents, ax=ax)

        for group_idx, group_name in enumerate(group_names):
            _plot_group_outliers(ax_data, plot_extents, group_idx=group_idx, group_name=group_name, ax=ax)

    _add_to_kwargs('x', x), _add_to_kwargs('y', y), _add_to_kwargs('hue', hue), _add_to_kwargs('data', data)
    value_label, group_label, group_names, orient, hue_label, plot_data = _get_info()
    plot_data: np.array = _get_plot_data()

    actual_data: Union[pd.Series, pd.DataFrame, np.ndarray] = plot_data if data is None else data
    actual_data: pd.Series = pd.Series(actual_data) if type(actual_data) not in (
        pd.Series, pd.DataFrame) else actual_data

    cutoff_lo, cutoff_hi = _get_cutoffs(plot_data)

    inlier_data = _get_inlier_data(actual_data, plot_data, cutoff_lo, cutoff_hi)
    kwargs[_which_data_var()] = inlier_data

    plot = plotter(**kwargs)

    if type(plot) == sns.FacetGrid:
        sample_ax = np.hstack(plot.axes)[0]
        plot_extents = np.array([sample_ax.get_xlim(), sample_ax.get_ylim()])

        row_names, col_names = [[None] if not a else a for a in [plot.row_names, plot.col_names]]
        for row, row_name in enumerate(row_names):
            row_df = actual_data if row_name is None else actual_data[actual_data[kwargs['row']] == row_name]
            for col, col_name in enumerate(col_names):
                ax_df = row_df if col_name is None else row_df[row_df[kwargs['col']] == col_name]
                _plot_ax_outliers(ax=plot.axes[row][col], ax_data=ax_df, plot_extents=plot_extents)

        for ax in np.hstack(plot.axes):
            _add_margins(ax, plot_data, cutoff_lo, cutoff_hi, orient, margin)
        return plot

    plot_extents = np.array([plot.get_xlim(), plot.get_ylim()])
    _plot_ax_outliers(plot, ax_data=actual_data, plot_extents=plot_extents)

    _add_margins(plot, plot_data, cutoff_lo, cutoff_hi, orient, margin)

    return plot
