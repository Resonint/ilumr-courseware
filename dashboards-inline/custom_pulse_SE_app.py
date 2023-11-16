import numpy as np
from collections import OrderedDict
import panel as pn
from time import time
from os import path
import yaml

from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
from matipo.util.autophase import autophase
from matipo.util.decimation import decimate
from matipo.util.plots import ComplexPlot
from matipo.util.fft import get_freq_spectrum
from matipo.util.dashboardapp import DashboardApp, ScaledInput

from util import get_current_values

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

POST_DEC = 4

DIR_PATH = path.dirname(path.realpath(__file__)) # directory of this file

def autophase_max(data):
    i_max = np.argmax(np.abs(data))
    phi = np.angle(data[i_max])
    return data*np.exp(-1j*phi)

class CustomPulseSEApp(DashboardApp):
    def __init__(self, override_pars={}, override_par_files=[], show_magnitude=False, show_complex=True, enable_run_loop=False, flat_filter=False):
        super().__init__(None, Sequence(path.join(DIR_PATH, 'programs/custom_pulse_SE.py')), enable_run_loop=enable_run_loop)
        
        with open(path.join(GLOBALS_DIR, 'gradient_calibration.yaml'), 'r') as f:
            self.gradient_calibration = float(yaml.load(f, Loader=yaml.SafeLoader)['gradient_calibration'])
            log.debug(f'gradient_calibration: {self.gradient_calibration}')
        
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
            x_axis_label="freq. encode axis (m)",
            y_axis_label="signal (arb units)")
        
        self.plot_row = pn.Row(self.plot1.figure, self.plot2.figure, sizing_mode='stretch_both')
        
        self.override_pars = override_pars
        self.override_par_files = override_par_files
        self.flat_filter = flat_filter
    
    def progress_handler(self, p, l):
        self.progress.max = int(l)
        self.progress.value = int(p)
        pn.io.push_notebook(self.progress)
    
    async def run(self):
        self.seq.loadpar(path.join(GLOBALS_DIR, 'hardpulse_90.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'hardpulse_180.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'frequency.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'shims.yaml'))
        
        for filename in self.override_par_files:
            self.seq.loadpar(filename)
        
        self.seq.setpar(**get_current_values(self.override_pars))
        
        if self.flat_filter:
            # adjust n_samples and t_dw for postprocess decimation
            self.seq.setpar(
                n_samples=POST_DEC*self.seq.par.n_samples,
                t_dw=self.seq.par.t_dw/POST_DEC)

        log.debug(self.seq.par)
        await self.seq.run(progress_handler=self.progress_handler)
        t0 = self.seq.par.t_dw*self.seq.par.n_samples/2
        t_dw = self.seq.par.t_dw
        n_samples = self.seq.par.n_samples
        y = self.seq.data
        if self.flat_filter:
            t_dw *= POST_DEC
            n_samples//=POST_DEC
            y = decimate(y, POST_DEC)
        
        
        t0 = t_dw*(n_samples)/2
        # y = autophase(y, t0=t0, dwelltime=t_dw)
        y = autophase_max(y) # use simpler autophase that works better for imaging
        x = np.linspace(0, n_samples*t_dw, n_samples, endpoint=False)
        
        freq, fft = get_freq_spectrum(y, t_dw)
        fft *= np.exp(1j * 2 * np.pi * -t0 * freq)  # correct for time shift
        spatial_axis = freq*self.gradient_calibration/np.linalg.norm(self.seq.par.g_read)
        
        self.plot1.update_data(x, y)
        self.plot2.update_data(spatial_axis, fft)
        pn.io.push_notebook(self.plot_row)
    
    def main(self):
        return pn.Column(
            self.plot_row,
            self.control_row,
            sizing_mode='stretch_both')
