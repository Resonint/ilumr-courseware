import panel as pn
import sys
import os
import yaml
import numpy as np

from matipo import Sequence, SEQUENCE_DIR, GLOBALS_DIR, DATA_DIR

from matipo.experiment.base_experiment import PlotInterface
from matipo.experiment.models import PLOT_COLORS, SITickFormatter, SIHoverFormatter

from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool,LabelSet

from matipo.util.decimation import decimate
from matipo.util.fft import fft_reconstruction
from matipo.util.etl import deinterlace

class ROIStatsImagePlot(PlotInterface):
    def __init__(self, figure_opts={}, image_opts={}, roi_circle_opts={}, roi_label_opts={},color_palette=PLOT_COLORS):
        _figure_opts = dict(
            title='Image',
            x_axis_label='x (mm)',
            y_axis_label='y (mm)',
            sizing_mode='stretch_both',
            min_height=400,
            min_width=400,
            match_aspect=True,
            aspect_scale=1,
            toolbar_location="above",
            tools='pan,wheel_zoom,box_zoom,reset,save'
        )
        _figure_opts.update(figure_opts) # allow options to be overriden
        
        self._image_opts = dict(
            palette='Greys256'
        )
        self._image_opts.update(image_opts) # allow options to be overriden
        
        self._roi_circle_opts= dict(
            fill_alpha=0,
            # color='red'
        ) 
        self._roi_circle_opts.update(roi_circle_opts)
        self._roi_label_opts= dict(
            text_align='center',
            x_offset=0,
            y_offset=-40,
            text_color='white',
            text_alpha=1.0,
            background_fill_color='black',
            background_fill_alpha=0.2
        ) 
        self._roi_label_opts.update(roi_label_opts)
        
        self._color_palette=color_palette
        
        self.image_data_source = None
        self.roi_data_source = None

        self.fig = figure(**_figure_opts)
        self.fig.toolbar.logo = None
        self.fig.toolbar.active_drag = self.fig.tools[2]
        

    def update(self, kdata, fov, roi_center_x, roi_center_y, roi_radius):
        # data processing/reconstruction
        imdata = np.abs(fft_reconstruction(kdata, gaussian_blur=1, upscale_factor=2))
        
        x_mm = np.linspace(-fov/2,fov/2,imdata.shape[0])
        y_mm = np.linspace(-fov/2,fov/2,imdata.shape[1])
        mean = []
        std = []
        label = []
        
        for x,y,r in zip(roi_center_x,roi_center_y,roi_radius):
            distsq = (x_mm[np.newaxis,:]-x)**2+(y_mm[:,np.newaxis]-y)**2
            mask = distsq < r**2
            _mean = np.mean(imdata[mask])
            mean.append(_mean)
            _std = np.std(imdata[mask])
            std.append(_std)
            label.append(f'μ={_mean:.2e}\nσ={_std:.2e}')
            
        rep = int(np.ceil(len(roi_center_x)/len(self._color_palette)))
        colors=rep*self._color_palette
        
        self.state = dict(
            image=dict(
                d=[imdata],
                x=[-(fov/2)],
                y=[-(fov/2)],
                dw=[fov],
                dh=[fov]),
            roi=dict(
                x=roi_center_x,
                y=roi_center_y,
                r=roi_radius,
                label_y=np.array(roi_center_y)-np.array(roi_radius),
                mean=mean,
                std=std,
                label=label,
                colors=colors[:len(roi_center_x)])
        )
    
    @property
    def roi_mean(self):
        return self.roi_data_source.data['mean']
    
    @property
    def roi_std(self):
        return self.roi_data_source.data['std']
    
    @property
    def state(self):
        """Should override to return plot state"""
        return dict(
            image=self.image_data_source.data,
            roi=self.roi_data_source.data)
    
    @state.setter
    def state(self, data):
        """Should override to apply plot state"""
        
        if self.image_data_source is None:
            self.image_data_source = ColumnDataSource(data=data['image'])
            self.image = self.fig.image(image='d', x='x', y='y', dw='dw', dh='dh', source=self.image_data_source, **self._image_opts)
            self.image_hover = HoverTool(
                tooltips=[
                    ('value:', '@d'),
                    ('(x, y):', '($x, $y)')
                ], 
                renderers=[self.image])
            self.fig.add_tools(self.image_hover)
        else:
            self.image_data_source.data = data['image']
        
        if self.roi_data_source is None:
            self.roi_data_source = ColumnDataSource(data=data['roi'])
            self.roi = self.fig.circle(x='x', y='y', radius='r',color='colors', source=self.roi_data_source, **self._roi_circle_opts) 
            self.roi_labels = LabelSet(x='x', y='label_y', text='label', source=self.roi_data_source, **self._roi_label_opts)
            self.fig.add_layout(self.roi_labels)
        else:
            self.roi_data_source.data = data['roi']

    def __call__(self):
        return self.fig
    