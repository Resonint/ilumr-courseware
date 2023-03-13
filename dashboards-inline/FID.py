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

class FIDApp(DashboardApp):
    def __init__(self, override_pars={}, override_par_files=[], show_magnitude=False, show_complex=True, enable_run_loop=False):
        super().__init__(None, Sequence(path.join(SEQUENCE_DIR, 'FID.py')), enable_run_loop=enable_run_loop)
        
        self.plot1 = ComplexPlot(
            title="Signal",
            show_magnitude=show_magnitude,
            show_complex=show_complex,
            x_axis_label="time (s)",
            y_axis_label="signal (V)")
        self.plot2 = ComplexPlot(
            title="Spectrum",
            show_magnitude=show_magnitude,
            show_complex=show_complex,
            x_axis_label="relative frequency (Hz)",
            y_axis_label="spectral density (V/kHz)")
        
        self.plot_row = pn.Row(self.plot1.figure, self.plot2.figure, sizing_mode='stretch_both')
        
        self.override_pars = override_pars
        self.override_par_files = override_par_files
    
    async def run(self):
        self.seq.loadpar(path.join(GLOBALS_DIR, 'hardpulse_90.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'frequency.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'shims.yaml'))
        
        for filename in self.override_par_files:
            self.seq.loadpar(filename)
        
        self.seq.setpar(**get_current_values(self.override_pars))

        log.debug(self.seq.par)
        await self.seq.run()
        y = autophase(self.seq.data)
        x = np.linspace(0, self.seq.par.n_samples*self.seq.par.t_dw, self.seq.par.n_samples)
        freq, fft = get_freq_spectrum(y, self.seq.par.t_dw)
        self.plot1.update_data(x, y)
        self.plot2.update_data(freq, fft)
        pn.io.push_notebook(self.plot_row)
    
    def main(self):
        return pn.Column(
            self.plot_row,
            self.control_row,
            sizing_mode='stretch_both')
