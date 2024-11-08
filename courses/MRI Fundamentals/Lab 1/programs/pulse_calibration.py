import numpy as np
from collections import OrderedDict
import panel as pn
from time import time
from os import path

from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
from matipo.util.autophase import autophase
from matipo.util.plots import ComplexPlot, SharedXPlot
from matipo.util.fft import get_freq_spectrum
from matipo.util.dashboardapp import DashboardApp, ScaledInput

from util import get_current_values

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


def gaussian_apodize(data, lb):
    t = np.linspace(0, 1, len(data))
    return data * np.exp(-lb*lb*t*t)


class PulseCalibrationApp(DashboardApp):
    def __init__(self, amplitude_steps=51, override_pars={}, override_par_files=[], show_magnitude=False, show_complex=True, enable_run_loop=False):
        super().__init__(None, Sequence(path.join(SEQUENCE_DIR, 'FID.py')), enable_run_loop=enable_run_loop)
        
        self.amplitude_steps = amplitude_steps
        
        self.plot1 = ComplexPlot(
            title="Signal",
            show_magnitude=show_magnitude,
            show_complex=show_complex,
            x_axis_label="time (s)",
            y_axis_label="signal (V)")
        
        self.plot2_data = {'x': [0], 'ys': {'sumsq': [0]}}
        self.plot2 = SharedXPlot(
            self.plot2_data['x'],
            self.plot2_data['ys'],
            title="Pulse Amplitude Calibration",
            x_axis_label="Pulse Amplitude (%)",
            y_axis_label="Signal Sum of Squares (V²)")
        
        self.plot_row = pn.Row(self.plot1.figure, self.plot2.figure, sizing_mode='stretch_both')
        
        self.override_pars = override_pars
        self.override_par_files = override_par_files
    
    async def run(self):
        self.plot2_data['x'] = [0]
        self.plot2_data['ys'] = {'sumsq': [0]}
        self.plot2.update(self.plot2_data['x'], self.plot2_data['ys'])
        pn.io.push_notebook(self.plot_row)
        
        self.seq.loadpar(path.join(GLOBALS_DIR, 'hardpulse_90.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'frequency.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'shims.yaml'))
        
        for filename in self.override_par_files:
            self.seq.loadpar(filename)
        
        self.seq.setpar(**get_current_values(self.override_pars))
        
        a_90 = np.linspace(0, 1, self.amplitude_steps, endpoint=True)

        log.debug(self.seq.par)
        
        for i, a_90_i in enumerate(a_90):
            self.seq.setpar(a_90=a_90_i)
            await self.seq.run()
            y = autophase(self.seq.data, fast=True)
            x = np.linspace(0, self.seq.par.n_samples*self.seq.par.t_dw, self.seq.par.n_samples)
            
            y_sumsq = np.sum(gaussian_apodize(np.abs(y), 2) ** 2)
            
            if i==0:
                self.plot2_data['x'][0] = a_90_i*100
                self.plot2_data['ys']['sumsq'][0] = y_sumsq
            else:
                self.plot2_data['x'].append(a_90_i*100)
                self.plot2_data['ys']['sumsq'].append(y_sumsq)
            
            self.plot1.update_data(x, y)
            self.plot2.update(self.plot2_data['x'], self.plot2_data['ys'])
            self.set_progress(i, len(a_90))
            pn.io.push_notebook(self.plot_row)
    
    def main(self):
        return pn.Column(
            self.plot_row,
            self.control_row,
            sizing_mode='stretch_both')
