import numpy as np
from collections import OrderedDict
import panel as pn
import asyncio
from os import path

from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
from matipo.util.plots import WobblePlot
from matipo.util.dashboardapp import DashboardApp, ScaledInput

from util import get_current_values

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


class WobbleApp(DashboardApp):
    def __init__(self, override_pars={}):
        super().__init__(None, Sequence(path.join(SEQUENCE_DIR, 'wobble.py')), enable_progressbar=False, enable_run_loop=True)
        
        self.plot = WobblePlot(
            title="Reflected Signal",
            x_axis_label="frequency (Hz)",
            y_axis_label="",
            y_axis_type='log')
        
        self.plot.figure.hover.tooltips = [("", "(@x, @y)")]
        
        self.plot_row = pn.Row(self.plot.figure, sizing_mode='stretch_both')
        self.control_row = pn.Row(*self.control_list, sizing_mode='stretch_width')
        
        self.override_pars = override_pars
    
    async def run(self):
        self.seq.loadpar(path.join(GLOBALS_DIR, 'frequency.yaml'))
        
        self.seq.setpar(**get_current_values(self.override_pars))
        
        # save display_f for plotting the marker
        display_f = self.seq.par.f
        # if the frequency range would overlap 0, shift the centre frequency to avoid negative frequencies
        if self.seq.par.f < self.seq.par.f_bw/2:
            self.seq.setpar(f=self.seq.par.f_bw/2)
        
        log.debug(self.seq.par)
        await self.seq.run()
        y = np.abs(np.mean(self.seq.data.reshape(-1, self.seq.par.n_samples), axis=1))
        x = np.linspace(self.seq.par.f - self.seq.par.f_bw/2, self.seq.par.f + self.seq.par.f_bw/2, len(y))
        self.plot.update_data(x, y, display_f)
        pn.io.push_notebook(self.plot_row)
        # await asyncio.sleep(0.5) # avoid running too fast in run loop, as the UI can become slow to respond
        
    
    def main(self):
        log.info('serving')
        return pn.Column(
            self.plot_row,
            self.control_row,
            sizing_mode='stretch_both')
