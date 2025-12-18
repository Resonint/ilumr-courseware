from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from collections import namedtuple

PARDEF = [
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('f_bw', float, 0.1e6, unit='Hz'),
    ParDef('n_steps', int, 512, min=1, unit=''),
    ParDef('t_dw', float, 8e-6, min=0.1e-6, max=80e-6, unit='s'),
    ParDef('n_samples', int, 4, min=2, unit='')
]

ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(p: ParameterSet):
    return seq.Options(
        amp_enabled=False,
        rx_gain=7)


def get_datalayout(p: ParameterSet):
    return datalayout.Repetitions(
        p.n_steps,
        datalayout.Acquisition(
            n_samples=p.n_samples,
            t_dw=p.t_dw))


def main(p: ParameterSet):
    t_acq = (p.n_samples+7) * p.t_dw
    f_start = p.f - p.f_bw/2
    f_step = p.f_bw/p.n_steps
    
    a_sweep = 0.25
    
    # for wobble don't execute a pulse_start so that the amp and duplexer switch aren't enabled (leaky abstraction?)
    for i in range(p.n_steps):
        f_sweep = f_start + i*f_step
        yield seq.pulse_update(f_sweep, 0, a_sweep)
        yield seq.wait(1e-6)
        yield seq.acquire(f_sweep, 0, p.t_dw, p.n_samples)
        yield seq.wait(t_acq)
    yield seq.pulse_end()
    yield seq.wait(1e-6)
