import numpy as np

from scipy.fft import fft,fftfreq, fftshift, ifftshift
from matipo.experiment.base_experiment import BaseExperiment
from matipo.experiment.plots import ComplexDataLinePlot


class CustomRF(BaseExperiment):
    
    def setup(self):
        self.plots = {
            'waveform': ComplexDataLinePlot(
                figure_opts = dict(
                    title="Waveform",
                    x_axis_label="Time (s)",
                    y_axis_label="Amplitude")),
            'spectrum': ComplexDataLinePlot(
                figure_opts = dict(
                    title="Spectrum",
                    x_axis_label="Frequency (Hz)",
                    y_axis_label="Spectral Density"))
        }
        
        self.pulse_shape = [1]
        self.pulse_width = 100e-6
        
    def calc_pulse_shape(self):
        N = 1
        dt = 100e-6
        pts = [1]
        width = dt*N
        return width, pts
        
    def pulse_settings_handler(self, event):
        self.pulse_width, self.pulse_shape = self.calc_pulse_shape()
        N = len(self.pulse_shape)
        dt = self.pulse_width/N
        t = np.arange(N)*dt
        t_0 = (self.pulse_width-dt)/2
        freq = fftshift(fftfreq(N,dt))
        spectrum = fftshift(fft(self.pulse_shape))*dt
        spectrum *= np.exp(-1j*np.pi*(self.pulse_width+dt)*freq) # fix phase of spectrum plot due to time offset
        self.plots['waveform'].update(t, self.pulse_shape)
        self.plots['spectrum'].update(freq, spectrum)
        
    def layout_controls(self):
        """Optionally override to change controls layout"""
        return None # self.log_handler.view