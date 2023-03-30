import numpy as np
from collections import OrderedDict
import panel as pn
from time import time
from os import path
import asyncio

from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
from matipo.util.decimation import decimate
from matipo.util.autophase import autophase
from matipo.util.plots import ComplexPlot, ImagePlot
from matipo.util.fft import get_freq_spectrum, fft_reconstruction
from matipo.util.dashboardapp import DashboardApp, ScaledInput

from util import get_current_values

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

DIR_PATH = path.dirname(path.realpath(__file__)) # directory of this file
POST_DEC = 4

class RARE2DApp(DashboardApp):
    def __init__(self, override_pars={}, override_par_files=[], show_magnitude=False, show_complex=True, enable_run_loop=False):
        super().__init__(None, Sequence(path.join(DIR_PATH, 'programs/RARE.py')), enable_run_loop=enable_run_loop)
        
        
        self.plot1 = ImagePlot(
            title="K-Space")
        self.plot2 = ImagePlot(
            title="Image")
        
        self.plot_row = pn.Row(self.plot1.figure, self.plot2.figure, sizing_mode='stretch_both')
        
        self.override_pars = override_pars
        self.override_par_files = override_par_files
        
        self.kdata_order = None
        
        
    
    async def replot(self):
        await self.seq.fetch_data()
        kdata = decimate(self.seq.data.reshape(-1, self.seq.par.n_samples), POST_DEC, axis=1)
        kdata = kdata[self.kdata_order]
        imdata = fft_reconstruction(kdata)
        self.plot1.update_data(np.abs(kdata), 1, 1) # TODO kspace axis scales
        self.plot1.im.glyph.color_mapper.low = 0
        self.plot2.update_data(np.abs(imdata), 1, 1) # TODO image axis scales
        pn.io.push_notebook(self.plot_row)
    
    def progress_handler(self, p, l):
        self.progress.max = int(l)
        self.progress.value = int(p)
        # pn.io.push_notebook(self.progress)
        asyncio.ensure_future(self.replot())
    
    async def run(self):
        self.seq.loadpar(path.join(GLOBALS_DIR, 'hardpulse_90.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'hardpulse_180.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'frequency.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'shims.yaml'))
        
        for filename in self.override_par_files:
            self.seq.loadpar(filename)
        
        self.seq.setpar(**get_current_values(self.override_pars))
        
        # adjust n_samples and t_dw for postprocess decimation
        self.seq.setpar(
            n_samples=POST_DEC*self.seq.par.n_samples,
            t_dw=self.seq.par.t_dw/POST_DEC)
        
        self.kdata_order = np.argsort(np.sum(self.seq.par.g_phase_1, axis=1), axis=0)
        
        log.debug(self.seq.par)
        await self.seq.run(progress_handler=self.progress_handler)
        # self.replot()
    
    def main(self):
        return pn.Column(
            self.plot_row,
            self.control_row,
            sizing_mode='stretch_both')
