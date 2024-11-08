import os
import yaml
import numpy as np

from matipo import Sequence, SEQUENCE_DIR, GLOBALS_DIR, DATA_DIR, Unit
from matipo.util.decimation import decimate
from matipo.util.fft import fft_reconstruction
from matipo.util.etl import deinterlace

from matipo.experiment.base_experiment import BaseExperiment
from matipo.experiment.plots import ImagePlot

GAMMA_BAR = 42.58e6

class Base2DRAREExperiment(BaseExperiment):
    def setup(self):
        self.seq = Sequence(SEQUENCE_DIR+'RARE.py')
        self.enable_partialplot = True
        
        self.plots = {
            'kspace': ImagePlot(
                figure_opts=dict(
                    title='k-space',
                    x_axis_label=r'Freq. Encode Axis (m^-1)',
                    y_axis_label=r'Phase. Encode Axis (m^-1)'
                ),
                image_opts=dict(palette='Viridis256')
            ),
            'image': ImagePlot(
                figure_opts=dict(
                    x_axis_label='Freq. Encode Axis (mm)',
                    y_axis_label='Phase. Encode Axis (mm)'
                )
            )
        }

        with open(os.path.join(GLOBALS_DIR, 'gradient_calibration.yaml'), 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            self.G_CAL = 1/(GAMMA_BAR*data['gradient_calibration']) # convert
            self.log.debug(f'G_CAL: {self.G_CAL}')

        self.DEC = 4 # postprocess decimation factor
        self.FOV = 10e-3 # m
        
        self.postproc = {
            'blur': 1,
            'upscale': 2
        }

        # set nice default pars
        self.seq.setpar(
            n_scans=4,
            t_end=0.5,
            n_phase_1 = 1,
            n_phase_2 = 32,
            g_phase_2 = (0, 0.5, 0),
            g_read = (0.5, 0, 0),
            t_dw=20e-6/self.DEC,
            n_samples=32*self.DEC,
            t_phase=320e-6
        )        

    def g_read_mag(self, fov, gamma_bar=GAMMA_BAR):
        """ Returns read gradient required for a given FOV """
        dt = self.seq.par.t_dw*self.DEC
        return max(0, min(1, 1/(self.G_CAL*fov*gamma_bar*dt))) # clip to range 0 to 1

    def g_phase_mag(self, fov, gamma_bar=GAMMA_BAR):
        """ Returns phase gradient required for a given FOV """
        dt = self.seq.par.t_phase/(self.seq.par.n_phase_2//2)
        return max(0, min(1, 1/(self.G_CAL*fov*gamma_bar*dt))) # clip to range 0 to 1

    def fov_read(self, gamma_bar=GAMMA_BAR):
        """ Returns read encode direction FOV """
        dt = self.seq.par.t_dw*self.DEC
        g = self.G_CAL*np.linalg.norm(self.seq.par.g_read)
        return 1/(g*gamma_bar*dt)

    def fov_phase(self, gamma_bar=GAMMA_BAR):
        """ Returns phase encode direction FOV """
        dt = self.seq.par.t_phase/(self.seq.par.n_phase_2//2)
        g = self.G_CAL*np.linalg.norm(self.seq.par.g_phase_2)
        return 1/(g*gamma_bar*dt)
    
    def update_par(self):
        # set calculated pars
        self.seq.setpar(
            n_ETL=self.seq.par.n_phase_2,
            t_read=self.seq.par.t_dw*self.seq.par.n_samples
        )
        self.seq.setpar(
            t_phase=self.seq.par.t_read/2
        )
        g_read_dir = self.seq.par.g_read/np.linalg.norm(self.seq.par.g_read)
        g_phase_dir = self.seq.par.g_phase_2/np.linalg.norm(self.seq.par.g_phase_2)
        self.seq.setpar(
            g_read = self.g_read_mag(self.FOV)*g_read_dir,
            g_phase_2 = self.g_phase_mag(self.FOV)*g_phase_dir
        )

    async def update_plots(self, final):
        await self.seq.fetch_data()
        kdata = decimate(self.seq.data.reshape(-1, self.seq.par.n_samples), self.DEC, axis=1)
        imdata = fft_reconstruction(kdata, gaussian_blur=self.postproc['blur'], upscale_factor=self.postproc['upscale'])
        fov_read = self.fov_read()
        fov_phase = self.fov_phase()
        self.plots['kspace'].update(np.abs(kdata), 1/fov_read, 1/fov_phase)
        self.plots['image'].update(np.abs(imdata), 1e3*fov_read, 1e3*fov_phase)
