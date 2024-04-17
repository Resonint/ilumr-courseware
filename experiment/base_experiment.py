import asyncio
import re
import panel as pn
from packaging import version

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# remove margins from progressbar that cause scrollbars and alignment issues
PROGRESS_STYLESHEET = """
progress {
  height: 100%;
  margin: 5px 0px 0px 0px;
  border-radius: 3px;
}

progress::-webkit-progress-value {
  border-radius: 3px;
}
"""

OLD_PANEL = version.parse(pn.__version__) < version.parse('1.0.0')
if OLD_PANEL:
    pn.config.raw_css += [PROGRESS_STYLESHEET]

# Interface that plot objects should extend for BaseExperiment understand them
class PlotInterface():
    def update(self, *args, **kwargs):
        """Override to update the plot using the data"""
        raise NotImplementedError

    def __call__(self):
        """Override to return the panel/bokeh object for display"""
        raise NotImplementedError


async def cancel_and_wait(task, msg=None):
    task.cancel(msg)
    try:
        await task
    except asyncio.CancelledError:
        if asyncio.current_task().cancelled():
            raise


class BaseExperiment():
    """
    Base class for experiments that handles running the pulse sequence and plotting the results.

    Overridable Attributes:
    -----------------------
    seq : Sequence
        Must override. Sequence that will be run by default `run()` method.
    plots : dict
        Should override. Dictionary of plot objects with a `figure` attribute and `update()` function.
    title : string
        May override.
    enable_runloop : bool
        May override. Enables `Run Loop` button.
    enable_progressbar : bool
        May override. Enables progress bar.
    enable_partialplot : bool
        May override. Enables plotting partial data on progress event.

    Overridable Methods:
    --------------------
    update_par
        Should override. Will be called before every run. May be using to update sequence parameters.
    update_plots
        Should override. Will be called after every run or progress event. May be used to update plots.
    layout_title
        May override.
    layout_plots
        May override.
    layout_controls
        May override.
    layout_app
        May override.
    run
        May override.
    abort
        May override.
    """

    # public attributes
    title = None
    seq = None
    plots = {}
    enable_runloop = False
    enable_progressbar = True
    enable_partialplot = False  # TODO: need to run seq.fetch_data()
    partialplot_cooldown = 1.0

    # private attributes
    _aborted = False
    _runtask = None
    _updateplotstask = None

    def __init__(self):
        if self.title is None:
            # use class name by default, adding spaces
            self.title = re.sub(r'(?<!^)(?=[A-Z])', ' ', self.__class__.__name__)
        
        self.run_btn = pn.widgets.Button(
            name='Run',
            button_type='success',
            align='end',
            sizing_mode='stretch_width')
        self.run_btn.on_click(self._run_handler)
        
        self.runloop_btn = pn.widgets.Button(name='Run Loop', button_type='success', align='end', sizing_mode='stretch_width')
        self.runloop_btn.on_click(self._runloop_handler)

        self.abort_btn = pn.widgets.Button(name='Abort', button_type='danger', align='end', sizing_mode='stretch_width')
        self.abort_btn.on_click(self._abort_handler)
        
        self.status = pn.widgets.TextInput(value='Idle', disabled=True, align='end', sizing_mode='stretch_width')
        
        self.progress = pn.widgets.Progress(name='Progress', value=0, height=32, margin=0, sizing_mode='stretch_width')
        if not OLD_PANEL:
            self.progress.stylesheets = [PROGRESS_STYLESHEET]

        self.controls = [self.run_btn, self.abort_btn, self.status]
        if self.enable_progressbar:
            self.controls.insert(2, self.progress)
        if self.enable_runloop:
            self.controls.insert(1, self.runloop_btn)
        self.control_container = self.layout_controls()

        self.plot_container = self.layout_plots()

        self.title_container = self.layout_title()

        self.app = self.layout_app()

    def layout_plots(self):
        """Optionally override to change plots layout"""
        # get list of figures from plot dictionary
        figlist = [plot() for key, plot in self.plots.items()]
        return pn.Row(*figlist, min_height=400, sizing_mode='stretch_both')

    def layout_controls(self):
        """Optionally override to change controls layout"""
        return pn.Row(*self.controls, sizing_mode='stretch_width')

    def layout_title(self):
        """Optionally override to change the title formatting"""
        return pn.pane.HTML(f'<h2>{self.title}</h2>', sizing_mode='stretch_width')

    def layout_app(self):
        """Optionally override to change app layout"""
        return pn.Column(
            self.title_container,
            self.plot_container,
            self.control_container,
            sizing_mode='stretch_both')
    
    def update_par(self):
        """Override this method to set parameter values before running"""
        pass

    async def update_plots(self):
        """Override this method to update plots"""
        pass

    async def _run_handler(self, e):
        self._aborted = False
        self.run_btn.disabled = self.runloop_btn.disabled = True
        self.progress.value = 0
        self.set_status('Running...')
        try:
            if self._runtask is None or self._runtask.done():
                self._runtask = asyncio.create_task(self._run_wrapper())
            else:
                raise Exception("Already Running!")
            
            await self._runtask
            self.progress.value = self.progress.max
            self.set_status('Finished')
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.set_status(str(e))
            log.exception(e)
        self.run_btn.disabled = self.runloop_btn.disabled = False

    async def _runloop_handler(self, e):
        self.run_btn.disabled = self.runloop_btn.disabled = True
        self._aborted = False
        while not self._aborted:
            self.set_status('Running Forever')
            self.progress.value = 0
            try:
                if self._runtask is None or self._runtask.done():
                    self._runtask = asyncio.create_task(self._run_wrapper())
                else:
                    raise Exception("Already Running!")
                
                await self._runtask
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.set_status(str(e))
                log.exception(e)
                break
            # throttle loop
            await asyncio.sleep(0.01)
        self.run_btn.disabled = self.runloop_btn.disabled = False

    def _abort_handler(self, e):
        log.debug('aborting')
        self.set_status('Aborting...')
        try:
            self._aborted = True
            self.abort()
            self._runtask.cancel()
            self.set_status('Aborted')
        except Exception as e:
            self.set_status(str(e))
            log.exception(e)
        self.run_btn.disabled = self.runloop_btn.disabled = False

    async def _run_wrapper(self):
        self.update_par()

        await self.run()

        # cancel any ongoing plot update
        if self._updateplotstask is not None:
            await cancel_and_wait(self._updateplotstask)

        await self._update_plots_wrapper()

    async def _update_plots_wrapper(self):
        # workaround for bokeh plots not updating when using "preview with panel"
        curdoc = pn.state.curdoc
        if curdoc is not None and curdoc.session_context is not None:
            curdoc.add_next_tick_callback(self.update_plots)
        else:
            await self.update_plots()
            pn.io.push_notebook(self.plot_container)
        await asyncio.sleep(self.partialplot_cooldown)

    def _progress_handler(self, p, l):
        self.progress.max = int(l)
        self.progress.value = int(p)
        #  plot partial data on all except first and last
        if self.enable_partialplot and 0 < p < l:
            # only update plots if a plot update is not already running
            if self._updateplotstask is None or self._updateplotstask.done():
                self._updateplotstask = asyncio.create_task(self._update_plots_wrapper())

    def set_status(self, status):
        self.status.value = status

    async def run(self):
        """Optionally override to change run logic"""
        await self.seq.run(progress_handler=self._progress_handler)

    def abort(self):
        """Optionally override to change abort logic"""
        self.seq.abort()

    def __call__(self):
        return self.app
    