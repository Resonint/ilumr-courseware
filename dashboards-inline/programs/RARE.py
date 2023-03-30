from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from collections import namedtuple
import numpy as np
from functools import partial
from matipo.util.pulseshape import calc_soft_pulse
from matipo.util import etl

# TODO: move to library
def float_array(v):
    a = np.array(v, dtype=float)
    a.setflags(write=False)
    return a

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

g_ZERO = float_array((0,0,0))

# TODO: move to library
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

# TODO: move to library
def gen_grad_ramp(g_start, g_end, t, n):
    g_step = (g_end - g_start)/n
    t_step = t/n
    ret = b''
    for i in range(n):
        ret += seq.gradient(*(g_start+(i+1)*g_step))
        ret += seq.wait(t_step)
    return ret

PARDEF = [
    ParDef('n_scans', int, 2, min=1),
    ParDef('f', float, 1e6),
    ParDef('a_90', float, 0),
    ParDef('t_90', float, 32e-6),
    ParDef('f_slice_offset', float, 0),
    ParDef('bw_90', float, 0),
    ParDef('a_180', float, 0),
    ParDef('t_180', float, 32e-6),
    ParDef('t_inv', float, 1e-3),
    ParDef('t_dw', float, 1e-6),
    ParDef('n_samples', int, 320),
    ParDef('t_read', float, 320e-6),
    ParDef('t_phase', float, 160e-6),
    ParDef('n_ETL', int, 0),
    ParDef('t_end', float, 1),
    ParDef('g_slice', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('g_read', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('g_phase_1', float_array, [(0, 0, 0)], min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('g_phase_2', float_array, [(0, 0, 0)], min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_spoil', float, 1e-3),
    ParDef('g_spoil', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_grad_ramp', float, 100e-6),
    ParDef('n_grad_ramp', int, 10),
    ParDef('tweak_unslice', float, 0.0),
    ParDef('p_90_offset', float, 0),
    ParDef('shim_x', float, 0, min=-1, max=1),
    ParDef('shim_y', float, 0, min=-1, max=1),
    ParDef('shim_z', float, 0, min=-1, max=1),
    ParDef('shim_z2', float, 0, min=-1, max=1),
    ParDef('shim_zx', float, 0, min=-1, max=1),
    ParDef('shim_zy', float, 0, min=-1, max=1),
    ParDef('shim_xy', float, 0, min=-1, max=1),
    ParDef('shim_x2y2', float, 0, min=-1, max=1),
    ParDef('enable_IR', bool, False),
    ParDef('enable_dummy_run', bool, False)
]

# TODO: move to library
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(p: ParameterSet):
    return seq.Options(
        amp_enabled=True,
        rx_gain=7)


def get_datalayout(p: ParameterSet):
    n_runs, n_echos = etl.sequence_format(p.n_ETL, p.g_phase_1.shape[0], p.g_phase_2.shape[0])
#     if n_runs==1:
#         # to allow progress counting on scan
#         return datalayout.Scans(
#             p.n_scans,
#             datalayout.Repetitions(
#                 n_echos,
#                 datalayout.Acquisition(
#                     n_samples=p.n_samples,
#                     t_dw=p.t_dw)))
    # note: if the total number of phase steps does not divide evenly into n_ETL
    # there will be additional data in the last run with no phase encoding
    # which may be ignored or exploited
    return datalayout.Repetitions(
        n_runs,
        datalayout.Scans(
            p.n_scans,
            datalayout.Repetitions(
                n_echos,
                datalayout.Acquisition(
                    n_samples=p.n_samples,
                    t_dw=p.t_dw))))

def main(p: ParameterSet):
    n_runs, n_echos = etl.sequence_format(p.n_ETL, p.g_phase_1.shape[0], p.g_phase_2.shape[0])
    
    log.debug(f"n_runs: {n_runs}, n_echos: {n_echos}")

    t_echo = p.t_180 + 4*p.t_grad_ramp + 2*p.t_phase + p.t_read
    
    log.debug(f"echo time: {t_echo}")
    
    t_phase_read = p.t_read/2 - p.t_grad_ramp/2
    log.debug(f"t_phase_read: {t_phase_read}")
    t_phase_read_180 = (t_echo - p.t_90 - p.t_180)/2 - t_phase_read - 3*p.t_grad_ramp
    log.debug(f"preread gradient to 180 time: {t_phase_read_180}")
    if t_phase_read_180 <= 0:
        raise Exception('phase gradient time too short')
    
    t_acq_delay = (p.t_read - p.n_samples*p.t_dw)/2
    
    # time from end of inversion pulse to start of excitation pulse
    t_inv_90 = p.t_inv - (p.t_90+p.t_180)/2
    
    n_phase_cycle = 4
    phase_cycle_90 = [0, 180, 0, 180]
    phase_cycle_180 = [90, 90, 270, 270]
    
    p_acq_inc = 0
    
    enable_softpulse = p.bw_90 > 0
    if enable_softpulse:
        t_inv_90 -= p.t_grad_ramp # keep t_inv setting the time between centres of pulses
        
        # calculate the the gradient amplitude required to match half the slice gradient area during the read prephasing gradient time
        g_unslice = -(1+p.tweak_unslice)*p.g_slice*0.5*(p.t_90+p.t_grad_ramp)/(t_phase_read+p.t_grad_ramp)
        log.debug(f'g_unslice: {str(g_unslice)}')

        log.debug('using softpulse')
        # to offset the slice, the RF pulses must have a frequency offset,
        # which introduces a phase change relative to the receive phase. 
        # This needs to be corrected (TODO)
#         phase_diff = -360*p.f_slice_offset*(0.5*p.t_90+p.t_grad_ramp)
#         log.debug(f'phase diff: {phase_diff}')
#         #corr_phase_cycle = (np.array(phase_cycle_90) + phase_diff) % 360
        corr_phase_cycle = (np.array(phase_cycle_90) + p.p_90_offset) % 360
        p_acq_inc = 360*p.f_slice_offset*(p.t_90+p.t_grad_ramp)
        log.debug(f'p_acq_inc: {p_acq_inc}')
        log.debug(f'corrected phase cycle: {str(corr_phase_cycle)}')
        softpulse_cycle = gen_soft_pulse_cycle(p.f+p.f_slice_offset, corr_phase_cycle, p.a_90, p.t_90, p.bw_90)
    else:
        g_unslice = g_ZERO
    
    # gradient duty cycle check TODO: implement duty cycle checks in driver
    t_cpmg = p.t_90/2 + t_echo/2 + p.g_phase_2.shape[0]*t_echo - p.t_180/2 + p.t_spoil
    if p.enable_IR:
        t_cpmg += p.t_180/2 + p.t_inv - p.t_90/2
    t_rep = t_cpmg + p.t_end
    grad_total_area = (
        np.abs(p.g_slice)*(p.t_90+p.t_grad_ramp)
        + np.abs(p.g_read+g_unslice)*(t_phase_read+p.t_grad_ramp)
        # + p.g_phase_2.shape[0]*2*(np.abs(p.g_phase_1)+np.abs(p.g_phase_2))*p.t_phase
        + p.t_phase*(p.g_phase_2.shape[0]*np.sum(np.abs(p.g_phase_1)) + p.g_phase_1.shape[0]*np.sum(np.abs(p.g_phase_2))) # potentially overestimating if gradients are oblique
        + n_runs*np.abs(p.g_slice)*p.t_180
        + np.abs(p.g_spoil)*p.t_spoil
    )
    grad_duty_cycle = grad_total_area/t_rep
    log.debug(f'gradient duty cycle: {str(grad_duty_cycle)}')
    if np.any(grad_duty_cycle > 0.3):
        raise Exception('Gradient duty cycle too high!')
    
    # pregenerate fixed ramps (optimisation)
    g_ramp_zero_to_slice = gen_grad_ramp(g_ZERO, p.g_slice, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_slice_to_zero = gen_grad_ramp(p.g_slice, g_ZERO, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_zero_to_read_unslice = gen_grad_ramp(g_ZERO, p.g_read+g_unslice, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_read_unslice_to_zero = gen_grad_ramp(p.g_read+g_unslice, g_ZERO, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_zero_to_read = gen_grad_ramp(g_ZERO, p.g_read, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_read_to_zero = gen_grad_ramp(p.g_read, g_ZERO, p.t_grad_ramp, p.n_grad_ramp)
    
    yield seq.shim(p.shim_x, p.shim_y, p.shim_z, p.shim_z2, p.shim_zx, p.shim_zy, p.shim_xy, p.shim_x2y2)
    yield seq.wait(0.01)
    
    if p.enable_dummy_run:
        p_90 = phase_cycle_90[-1]
        p_180 = phase_cycle_180[-1]
        if p.enable_IR:
            # inversion pulse
            yield seq.pulse_start(p.f, 0, p.a_180)
            yield seq.wait(p.t_180)
            yield seq.pulse_end()
            yield seq.wait(t_inv_90)

        if enable_softpulse:
            yield (
                g_ramp_zero_to_slice
                + softpulse_cycle[-1]
                + g_ramp_slice_to_zero
                + seq.pulse_update(p.f, 0, 0)
            )
        else:
            yield (
                seq.pulse_start(p.f, p_90, p.a_90)
                + seq.wait(p.t_90)
                + seq.pulse_end()
                + seq.wait(p.t_grad_ramp)
            )

        yield seq.wait(t_phase_read)
        yield seq.wait(t_phase_read_180)

        for i_echo in range(n_echos):
            if enable_softpulse:
                yield (
                    # refocusing pulse
                    g_ramp_zero_to_slice
                    + seq.pulse_start(p.f+p.f_slice_offset, p_180, p.a_180)
                    + seq.wait(p.t_180)
                    + seq.pulse_end()
                    + g_ramp_slice_to_zero
                )
            else:
                yield (
                    # refocusing pulse
                    seq.pulse_start(p.f+p.f_slice_offset, p_180, p.a_180)
                    + seq.wait(p.t_180)
                    + seq.pulse_end()
                )
            
            # phase blip (dummy)
            yield seq.wait(p.t_phase)
            yield seq.wait(p.t_grad_ramp)

            # readout (dummy)
            yield seq.wait(p.t_grad_ramp)
            yield seq.wait(t_acq_delay)
            yield seq.wait(p.t_read-t_acq_delay)
            yield seq.wait(p.t_grad_ramp)
                    
            # unphase blip (dummy)
            yield seq.wait(p.t_phase)
            yield seq.wait(p.t_grad_ramp)
            
        # spoiler pulse (dummy)
        yield seq.wait(p.t_spoil)
        yield seq.wait(2*p.t_grad_ramp)

        yield seq.wait(p.t_end)
    
    for i_run in range(n_runs):
        for i_scan in range(p.n_scans):
            p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
            p_180 = phase_cycle_180[i_scan % n_phase_cycle]
    
            if p.enable_IR:
                # inversion pulse
                yield seq.pulse_start(p.f, 0, p.a_180)
                yield seq.wait(p.t_180)
                yield seq.pulse_end()
                yield seq.wait(t_inv_90)

            if enable_softpulse:
                yield (
                    g_ramp_zero_to_slice
                    + softpulse_cycle[i_scan % n_phase_cycle]
                    + g_ramp_slice_to_zero
                    +seq.pulse_update(p.f, 0, 0)
                )
            else:
                yield (
                    seq.pulse_start(p.f, p_90, p.a_90)
                    + seq.wait(p.t_90)
                    + seq.pulse_end()
                    + seq.wait(p.t_grad_ramp)
                )

            yield (
                g_ramp_zero_to_read_unslice
                + seq.wait(t_phase_read)
                + g_ramp_read_unslice_to_zero
                + seq.wait(t_phase_read_180)
            )

            for i_echo in range(n_echos):
                # calculate interlaced phase index
                i_phase = i_echo * n_runs + i_run
                if i_phase < p.g_phase_1.shape[0]*p.g_phase_2.shape[0]:
                    # calculate gradient
                    i_phase_1 = i_phase % p.g_phase_1.shape[0]
                    i_phase_2 = i_phase // p.g_phase_1.shape[0]
                    g_phase = p.g_phase_1[i_phase_1] + p.g_phase_2[i_phase_2]
                else:
                    # special case when p.g_phase_1.shape[0]*p.g_phase_2.shape[0] does not divide evenly into p.n_ETL
                    # and there are extra echos
                    g_phase = g_ZERO

                if enable_softpulse:
                    yield (
                        # refocusing pulse
                        g_ramp_zero_to_slice
                        + seq.pulse_start(p.f+p.f_slice_offset, p_180, p.a_180)
                        + seq.wait(p.t_180)
                        + seq.pulse_end()
                        + g_ramp_slice_to_zero
                    )
                else:
                    yield (
                        # refocusing pulse
                        seq.pulse_start(p.f+p.f_slice_offset, p_180, p.a_180)
                        + seq.wait(p.t_180)
                        + seq.pulse_end()
                    )
                
                yield (
                    # phase blip
                    seq.gradient(*g_phase)
                    + seq.wait(p.t_phase)
                    + seq.gradient(*g_ZERO)
                    + seq.wait(p.t_grad_ramp)

                    # readout
                    + g_ramp_zero_to_read
                    + seq.wait(t_acq_delay)
                    + seq.acquire(p.f, (p_acq + i_scan*p_acq_inc)%360, p.t_dw, p.n_samples)
                    + seq.wait(p.t_read-t_acq_delay)
                    + g_ramp_read_to_zero
                    
                    # unphase blip
                    + seq.gradient(*(-g_phase))
                    + seq.wait(p.t_phase)
                    + seq.gradient(*g_ZERO)
                    + seq.wait(p.t_grad_ramp)
                )
            
            # spoiler pulse
            yield seq.gradient(*p.g_spoil)
            yield seq.wait(p.t_spoil)
            yield seq.gradient(*g_ZERO)
            yield seq.wait(2*p.t_grad_ramp)

            yield seq.wait(p.t_end)
