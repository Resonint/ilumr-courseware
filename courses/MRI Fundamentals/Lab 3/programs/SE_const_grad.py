from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from collections import namedtuple
import numpy as np
from functools import partial
from matipo.util.pulseshape import calc_soft_pulse

# TODO: move to library
def float_array(v):
    a = np.array(v, dtype=float)
    a.setflags(write=False)
    return a

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

def gen_soft_pulse_cycle(freq, phase_cycle, amp, width, bandwidth):
    N, dt, pts = calc_soft_pulse(width, bandwidth)
    log.debug(f'softpulse N: {N}, dt: {dt}')
    pts *= amp
    softpulse_cycle = []
    for phase in phase_cycle:
        softpulse = seq.pulse_start(freq, phase, pts[0]) + seq.wait(dt)
        for amp in pts[1:]:
            softpulse += seq.pulse_update(freq, phase, amp) + seq.wait(dt)
        softpulse += seq.pulse_end()
        softpulse_cycle.append(softpulse)
    return softpulse_cycle

PARDEF = [
    ParDef('n_scans', int, 1, min=1),
    ParDef('f', float, 1e6),
    ParDef('a_90', float, 0),
    ParDef('t_90', float, 32e-6),
    ParDef('a_180', float, 0),
    ParDef('t_180', float, 32e-6),
    ParDef('t_dw', float, 1e-6),
    ParDef('n_samples', int, 1000),
    ParDef('t_echo', float, 2000e-6),
    ParDef('t_end', float, 1),
    ParDef('g_read', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_grad_rise', float, 200e-6),
    ParDef('shim_x', float, 0, min=-1, max=1),
    ParDef('shim_y', float, 0, min=-1, max=1),
    ParDef('shim_z', float, 0, min=-1, max=1),
    ParDef('shim_z2', float, 0, min=-1, max=1),
    ParDef('shim_zx', float, 0, min=-1, max=1),
    ParDef('shim_zy', float, 0, min=-1, max=1),
    ParDef('shim_xy', float, 0, min=-1, max=1),
    ParDef('shim_x2y2', float, 0, min=-1, max=1)
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
    t_acq = par.n_samples * par.t_dw
    t_90_180 = (par.t_echo - par.t_90 - par.t_180)/2
    log.debug(f"90 to 180 time: {t_90_180}")
    if t_90_180 <= 0:
        raise Exception('Echo time too short, or 90 pulse too long.')
    t_180_acq = (par.t_echo - par.t_180 - t_acq)/2
    log.debug(f"180 to acq time: {t_180_acq}")
    if t_180_acq <= 0:
        raise Exception('Echo time too short, or acquisition too long.')
    
    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]
    
    yield seq.shim(par.shim_x, par.shim_y, par.shim_z, par.shim_z2, par.shim_zx, par.shim_zy, par.shim_xy, par.shim_x2y2)
    yield seq.wait(0.01)
    
    for i_scan in range(par.n_scans):
        p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
        p_180 = phase_cycle_180[i_scan % n_phase_cycle]
        
        yield seq.gradient(*par.g_read)
        yield seq.wait(par.t_grad_rise)
        
        yield seq.pulse_start(par.f, p_90, par.a_90)
        yield seq.wait(par.t_90)
        yield seq.pulse_end()
        
        yield seq.wait(t_90_180)
        yield seq.pulse_start(par.f, p_180, par.a_180)
        yield seq.wait(par.t_180)
        yield seq.pulse_end()
        yield seq.wait(t_180_acq)
        
        yield seq.acquire(par.f, p_acq, par.t_dw, par.n_samples)
        yield seq.wait(t_acq)
        yield seq.gradient(0, 0, 0)
        yield seq.wait(par.t_end)
