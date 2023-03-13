from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from collections import namedtuple
import numpy as np
from functools import partial
from matipo.util.pulseshape import calc_soft_pulse

PARDEF = [
    ParDef('n_scans', int, 1, min=1),
    ParDef('f', float, 1e6),
    ParDef('a_90', float, 0),
    ParDef('t_90', float, 32e-6),
    ParDef('a_180', float, 0),
    ParDef('t_180', float, 32e-6),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=80e-6),
    ParDef('n_samples', int, 1000),
    ParDef('t_echo', float, 200e-6),
    ParDef('t_end', float, 1),
    ParDef('shim_x', float, 0, min=-1, max=1),
    ParDef('shim_y', float, 0, min=-1, max=1),
    ParDef('shim_z', float, 0, min=-1, max=1),
    ParDef('shim_z2', float, 0, min=-1, max=1),
    ParDef('shim_zx', float, 0, min=-1, max=1),
    ParDef('shim_zy', float, 0, min=-1, max=1),
    ParDef('shim_xy', float, 0, min=-1, max=1),
    ParDef('shim_x2y2', float, 0, min=-1, max=1),
]

# TODO: move to library
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(par: ParameterSet):
    return seq.Options(
        amp_enabled=True,
        rx_gain=7)


def get_datalayout(p: ParameterSet):
    return datalayout.Scans(
        p.n_scans,
        datalayout.Acquisition(
            n_samples=p.n_samples,
            t_dw=p.t_dw))


def main(par: ParameterSet):
    
    # calculate some timing parameters for the sequence
    t_acq = par.n_samples * par.t_dw # acquisition duration
    
    # set t_90_180 to satisfy echo time constraint
    t_90_180 = (par.t_echo - par.t_90 - par.t_180)/2 # time delay between 90 and 180 pulses
    
    # calculate time for acquisition to finish after 180 pulse
    t_finish = t_acq - par.t_180 - t_90_180 - par.t_90
    
    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]
    
    yield seq.shim(par.shim_x, par.shim_y, par.shim_z, par.shim_z2, par.shim_zx, par.shim_zy, par.shim_xy, par.shim_x2y2)
    yield seq.wait(0.01)
    
    for i_scan in range(par.n_scans):
        p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
        p_180 = phase_cycle_180[i_scan % n_phase_cycle]
        
        yield seq.acquire(par.f, p_acq, par.t_dw, par.n_samples)
        yield seq.pulse_start(par.f, p_90, par.a_90)
        yield seq.wait(par.t_90)
        yield seq.pulse_end()
        
        yield seq.wait(t_90_180)
        
        yield seq.pulse_start(par.f, p_180, par.a_180)
        yield seq.wait(par.t_180)
        yield seq.pulse_end()
        
        # wait for acquisition to finish
        yield seq.wait(t_finish)
        
        # end delay to control repetition time
        yield seq.wait(par.t_end)
