import panel as pn
import sys
import os
import yaml
import numpy as np

from matipo import Sequence, SEQUENCE_DIR, GLOBALS_DIR, DATA_DIR
from matipo.util.plots import SharedXPlot

from experiment import BaseExperiment # load before pn.extension() for stylesheet changes on panel < 1.0
from experiment.plot import SignalPlot, SpectrumPlot
from experiment.base_experiment import PlotInterface

from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool, Circle
from matipo.util.plots import PLOT_COLORS, get_SI_tick_formatter, get_SI_hover_formatter


class MultiDotPlot(PlotInterface):
    def __init__(self, figure_opts={}, scatter_opts={}, color_palette=[]):
        self.scatters = dict()
        self.sources = dict()
        
        _figure_opts = dict(
            toolbar_location="above",
            tools='pan,wheel_zoom,box_zoom,crosshair,reset,save',
            sizing_mode="stretch_both",
            min_height=400,
            min_width=400
        )
        _figure_opts.update(figure_opts) # allow options to be overriden
        
        self._scatter_opts = dict(
        )
        self._scatter_opts.update(scatter_opts) # allow options to be overriden
        
        self._color_palette=color_palette
        
        self._count = 0

        self.fig = figure(**_figure_opts)
        self.fig.axis.formatter = get_SI_tick_formatter()
        self.fig.toolbar.logo = None

    def update(self, data):
        """ Update the plot with a data dict, e.g.:
        data = dict(
            label_1=dict(x=[0], y=[0]),
            label_2=dict(x=[0], y=[0])
        )
        """
        
        for key in data:
            if key in self.sources:
                self.sources[key].data = data[key]
            else:
                self.sources[key] = ColumnDataSource(data = data[key])
                self.scatters[key] = self.fig.circle(x='x', y='y', color=self._color_palette[self._count%len(self._color_palette)], source=self.sources[key], **self._scatter_opts)
                self._count+=1
        
    def __call__(self):
        return self.fig
