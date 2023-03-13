import numpy as np
from collections import OrderedDict
import panel as pn
import yaml
from os import path
import asyncio

from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
from matipo.util.autophase import autophase
from matipo.util.optimisation import nelder_mead_async
from matipo.util.plots import SharedXPlot
from matipo.util.fft import get_freq_spectrum
from matipo.util.dashboardapp import DashboardApp, ScaledInput

from util import get_current_values

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

AUTOSHIM_OPTIONS = {
        'Quick': {
            'load_previous': True,
            'N_O1': 3,
            'N_O2': 5,
            'O1_shim_step': 0.05,
            'O2_shim_step': 0.05,
            'O1_precision': 0.0005,
            'O2_precision': 0.0005,
            'max_iterations': 100,
            'pars': {
                'n_scans': 2,
                't_end': 0.5,
                'n_samples': 1000,
                't_dw': 32e-6
            }
        }, 
        'Fine': {
            'load_previous': True,
            'N_O1': 3,
            'N_O2': 5,
            'O1_shim_step': 0.05,
            'O2_shim_step': 0.05,
            'O1_precision': 0.0005,
            'O2_precision': 0.0005,
            'max_iterations': 1000,
            'pars': {
                'n_scans': 4,
                't_end': 0.5,
                'n_samples': 1000,
                't_dw': 64e-6
            }
        },
        'Coarse': {
            'load_previous': False,
            'N_O1': 3,
            'N_O2': 5,
            'O1_shim_step': 0.5,
            'O2_shim_step': 0.5,
            'O1_precision': 0.005,
            'O2_precision': 0.005,
            'max_iterations': 400,
            'pars': {
                'n_scans': 1,
                't_end': 0.5,
                'n_samples': 1000,
                't_dw': 8e-6
            }
        }
}

SHIM_ORDER = ['shim_x', 'shim_y', 'shim_z', 'shim_z2', 'shim_zx', 'shim_zy', 'shim_xy', 'shim_x2y2']

def gaussian_apodize(data, lb):
    t = np.linspace(0, 1, len(data))
    return data * np.exp(-lb*lb*t*t)

class AutoshimApp(DashboardApp):
    def __init__(self, output_dir=None, override_pars={}, override_par_files=[]):
        super().__init__(None, Sequence(path.join(SEQUENCE_DIR, 'FID.py')))
        
        with open(path.join(GLOBALS_DIR, 'bandwidth_calibration.yaml'), 'r') as f:
            self.bandwidth_calibration = float(yaml.safe_load(f)['bandwidth_calibration'])
            log.debug(f'bandwidth_calibration: {self.bandwidth_calibration}')
        
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = GLOBALS_DIR
        
        self.override_pars = override_pars
        self.override_par_files = override_par_files
        
        self.plotquality_data = {'x': [0], 'ys': {'sumsq': [0]}}
        self.plotquality = SharedXPlot(
            self.plotquality_data['x'],
            self.plotquality_data['ys'],
            title="Shim Quality",
            x_axis_label="Iteration",
            y_axis_label="Signal Sum of Squares (V²)",
            height=300)
        
        self.plotvalues_data = {key: [0] for key in SHIM_ORDER}
        self.plotvalues = SharedXPlot(
            self.plotquality_data['x'],
            self.plotvalues_data,
            title="Shim Values",
            x_axis_label="Iteration",
            y_axis_label="Shim Value (% of max)",
            x_range=self.plotquality.figure.x_range, # share x ranges
            height=300,)
        
        self.plot_row = pn.Column(self.plotvalues.figure, self.plotquality.figure, sizing_mode='stretch_both')
        
        self.input_method_select = pn.widgets.Select(options=[k for k in AUTOSHIM_OPTIONS], value='Coarse', height=31, sizing_mode='stretch_width')
        self.control_row = pn.Row(self.input_method_select, self.run_btn, self.abort_btn, self.progress, self.status, sizing_mode='stretch_width')
    
    async def evalfunc(self, try_shims, dummy=False):
        log.debug('Trying (X: %.4f, Y: %.4f, Z: %.4f, Z2: %.4f, ZX: %.4f, ZY: %.4f, XY: %.4f, X2Y2: %.4f)' % tuple(try_shims.tolist()))
        for i, shim_name in enumerate(SHIM_ORDER):
            self.seq.setpar(shim_name, try_shims[i])

        await self.seq.run()
        data = self.seq.data
        
        # return early if it's a dummy run
        if dummy:
            return 0
        
        y = gaussian_apodize(self.seq.data, 2)
        result = np.sum(np.abs(y) ** 2)
        
        # bit hackish, plot needs data to display properly
        # so we replace the first point if the x data is `[0]`
        # then append on subsequent iterations
        if self.plotquality_data['x'] == [0]:
            self.plotquality_data['x'][0] = 1
            self.plotquality_data['ys']['sumsq'][0] = result
            for i, shim_name in enumerate(SHIM_ORDER):
                self.plotvalues_data[shim_name] = [100*try_shims[i]]
        else:
            self.plotquality_data['x'].append(self.plotquality_data['x'][-1]+1)
            self.plotquality_data['ys']['sumsq'].append(result)
            for i, shim_name in enumerate(SHIM_ORDER):
                self.plotvalues_data[shim_name].append(100*try_shims[i])
        self.plotvalues.update(self.plotquality_data['x'], self.plotvalues_data)
        self.plotquality.update(self.plotquality_data['x'], self.plotquality_data['ys'])
        pn.io.push_notebook(self.plot_row)
        return -result
    
    def saveshims(self, shims):
        best_shims_dict = {shim_name: shims[i] for i, shim_name in enumerate(SHIM_ORDER)}
        with open(path.join(self.output_dir, 'shims.yaml'), 'w') as f:
            yaml.dump(best_shims_dict, f, default_flow_style=False)
        log.debug('shims saved')
    
    async def run(self):
        log.info('running')
        self.plotquality_data['x'] = [0]
        self.plotquality_data['ys'] = {'sumsq': [0]}
        self.plotvalues_data = {key: [0] for key in SHIM_ORDER}
        self.plotvalues.update(self.plotquality_data['x'], self.plotvalues_data)
        self.plotquality.update(self.plotquality_data['x'], self.plotquality_data['ys'])
        self.plotvalues.figure.legend.location = 'bottom_right'
        self.plotvalues.figure.legend.orientation = 'horizontal'
        self.plotquality.figure.legend.location = 'bottom_right'
        pn.io.push_notebook(self.plot_row)
        
        self.seq.loadpar(path.join(GLOBALS_DIR, 'hardpulse_90.yaml'))
        self.seq.loadpar(path.join(GLOBALS_DIR, 'frequency.yaml'))
        
        method_settings = AUTOSHIM_OPTIONS[self.input_method_select.value]
        N_O1 = method_settings['N_O1']
        N_O2 = method_settings['N_O2']
        init_step = [method_settings['O1_shim_step']]*N_O1 + [method_settings['O2_shim_step']]*N_O2
        precision = [method_settings['O1_precision']]*N_O1 + [method_settings['O2_precision']]*N_O2
        lower_bounds = [-1]*(N_O1+N_O2)
        upper_bounds = [1]*(N_O1+N_O2)
        for name, value in method_settings['pars'].items():
            self.seq.setpar(name, value)
        
        for filename in self.override_par_files:
            self.seq.loadpar(filename)
        
        self.seq.setpar(**get_current_values(self.override_pars))
        
        shims = np.zeros(N_O1+N_O2)
        if method_settings['load_previous']:
            with open(path.join(self.output_dir, 'shims.yaml'), 'r') as f:
                shims_dict = yaml.safe_load(f)
                for i, shim_name in enumerate(SHIM_ORDER):
                    shims[i] = shims_dict[shim_name]
        
        # do a dummy so that the first actual run has similar T1 relaxation effects
        await self.evalfunc(shims, dummy=True)
        
        i = 0
        best_shims = shims
        self.set_progress(0, method_settings['max_iterations'])
        async for r in nelder_mead_async(self.evalfunc, shims, x_lb=lower_bounds, x_ub=upper_bounds, max_iter=method_settings['max_iterations'], step=init_step, x_precision=precision):
            if not np.array_equal(r[0], best_shims):
                best_shims = r[0].tolist()
                log.debug('New Best Shims So Far: (X: %.4f, Y: %.4f, Z: %.4f, Z2: %.4f, ZX: %.4f, ZY: %.4f, XY: %.4f, X2Y2: %.4f)' % tuple(best_shims))
                self.saveshims(best_shims)
            self.set_progress(i, method_settings['max_iterations'])
            i += 1
    
    def main(self):
        log.info('serving')
        return pn.Column(
            self.plot_row,
            self.control_row,
            sizing_mode='stretch_both')
