import numpy as np
from collections import OrderedDict
import panel as pn
from time import time
from os import path

from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
from matipo.util.autophase import autophase
from matipo.util.plots import ComplexPlot
from matipo.util.fft import get_freq_spectrum
from matipo.util.dashboardapp import DashboardApp, ScaledInput

from util import get_current_values

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

DIR_PATH = path.dirname(path.realpath(__file__)) # directory of this file

class FullAcqSEApp(DashboardApp):
    def __init__(self, override_pars={}, override_par_files=[], show_magnitude=False, show_complex=True, enable_run_loop=False):
        super().__init__(None, Sequence(path.join(DIR_PATH, 'programs/full_acq_SE.py')), enable_run_loop=enable_run_loop)
        
        self.plot = ComplexPlot(
            title="Signal",
            show_magnitude=show_magnitude,
            show_complex=show_complex,
            x_axis_label="time (s)",
            y_axis_label="signal (V)")
        
        self.plot_row = pn.Row(self.plot.figure, sizing_mode='stretch_both')
        
        self.override_pars = override_pars
        self.override_par_files = override_par_files
    
    async def run(self):
        self.seq.loadpar(path.join(GLOBALS_DIR, 'hardpulse_90.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'hardpulse_180.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'frequency.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'shims.yaml'))
        
        for filename in self.override_par_files:
            self.seq.loadpar(filename)
        
        self.seq.setpar(**get_current_values(self.override_pars))

        log.debug(self.seq.par)
        await self.seq.run()
        y = self.seq.data
        x = np.linspace(0, self.seq.par.n_samples*self.seq.par.t_dw, self.seq.par.n_samples)
        self.plot.update_data(x, y)
        pn.io.push_notebook(self.plot_row)
    
    def main(self):
        return pn.Column(
            self.plot_row,
            self.control_row,
            sizing_mode='stretch_both')
