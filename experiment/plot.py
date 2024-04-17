# Plot Classes

import numpy as np

from matipo.util.plots import ComplexPlot
from matipo.util.autophase import autophase
from matipo.util.fft import get_freq_spectrum

from .base_experiment import PlotInterface

#
# Data Processing Functions
#

# TODO: (Optimisation) use memoization or other method to avoid recomputation

def process_signal(seqdata, t_dw):
    signal = autophase(seqdata)
    n_samples = len(signal)
    t = np.linspace(0, n_samples*t_dw, n_samples)
    return t, signal


def process_spectrum(seqdata, t_dw):
    t, signal = process_signal(seqdata, t_dw)
    freq, spectrum = get_freq_spectrum(signal, t_dw)
    return freq, spectrum

#
# Plot Implementations
#

class SignalPlot(PlotInterface):
    def __init__(self, **kwargs):
        options = dict(
            title="Signal",
            x_axis_label="time (s)",
            y_axis_label="signal (V)")
        options.update(kwargs) # allow options to be overriden
        self.plot_obj = ComplexPlot(**options)

    def update(self, seqdata, t_dw):
        self.plot_obj.update_data(*process_signal(seqdata, t_dw))

    def __call__(self):
        return self.plot_obj.figure


class SpectrumPlot(PlotInterface):
    def __init__(self, **kwargs):
        options = dict(
            title="Spectrum",
            x_axis_label="relative frequency (Hz)",
            y_axis_label="spectral density (V/kHz)")
        options.update(kwargs) # allow options to be overriden
        self.plot_obj = ComplexPlot(**options)

    def update(self, seqdata, t_dw):
        self.plot_obj.update_data(*process_spectrum(seqdata, t_dw))

    def __call__(self):
        return self.plot_obj.figure
