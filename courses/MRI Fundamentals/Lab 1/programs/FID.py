from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from matipo.util.pulseshape import calc_soft_pulse
from collections import namedtuple

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

def gen_soft_pulse_cycle(freq, phase_cycle, amp, width, bandwidth):
    N, dt, pts = calc_soft_pulse(width, bandwidth)
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
    ParDef('n_scans', int, 1, min=1, unit=''),
    ParDef('f', float, 1e6, min=0, unit='Hz'),
    ParDef('a_90', float, 0, min=0, max=1, unit=''),
    ParDef('t_90', float, 32e-6, min=0, unit='s'),
    ParDef('bw_90', float, 0, min=0, unit='Hz'),
    ParDef('t_acqdelay', float, 50e-6, min=0, unit='s'),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=80e-6, unit='s'),
    ParDef('n_samples', int, 1000, min=2, unit=''),
    ParDef('t_end', float, 1, min=0, unit='s'),
    ParDef('shim_x', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_y', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_z', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_z2', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_zx', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_zy', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_xy', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_x2y2', float, 0, min=-1, max=1, unit='')
]

ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(p: ParameterSet):
    return seq.Options(
        amp_enabled=True,
        rx_gain=7)


# TODO: generate automatically
def get_datalayout(p: ParameterSet):
    return datalayout.Scans(
        p.n_scans,
        datalayout.Acquisition(
            n_samples=p.n_samples,
            t_dw=p.t_dw))


def main(p: ParameterSet):
    t_acq = p.n_samples * p.t_dw
    
    n_phase_cycle = 4
    phase_cycle = [0, 180, 90, 270]
    
    enable_softpulse = p.bw_90 > 0
    if enable_softpulse:
        log.debug('using softpulse')
        softpulse_cycle = gen_soft_pulse_cycle(p.f, phase_cycle, p.a_90, p.t_90, p.bw_90)
    
    yield seq.shim(p.shim_x, p.shim_y, p.shim_z, p.shim_z2, p.shim_zx, p.shim_zy, p.shim_xy, p.shim_x2y2)
    yield seq.wait(0.01)
    
    for i in range(p.n_scans):
        phase = phase_cycle[i % n_phase_cycle]
        
        if enable_softpulse:
            yield softpulse_cycle[i % n_phase_cycle]
        else:
            yield seq.pulse_start(p.f, phase, p.a_90)
            yield seq.wait(p.t_90)
            yield seq.pulse_end()
        
        yield seq.wait(p.t_acqdelay)
        yield seq.acquire(p.f, phase, p.t_dw, p.n_samples)
        yield seq.wait(t_acq)
        
        yield seq.wait(p.t_end)
