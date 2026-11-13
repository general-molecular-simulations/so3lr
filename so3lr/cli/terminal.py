"""Import-cheap signal ownership for interactive terminal contexts only."""
from contextlib import contextmanager
import signal
import threading

# Set inside batch-scheduler jobs: no prompts, no network checks there.
SCHEDULER_VARIABLES = ('SLURM_JOB_ID', 'SLURM_JOBID', 'PBS_JOBID', 'LSB_JOBID',
                       'JOB_ID', 'COBALT_JOBID')


class TerminationSignal(BaseException):
    def __init__(self, number):
        self.number = number
        super().__init__(f'Terminated by {signal.Signals(number).name}')


@contextmanager
def input_mode(keyboard):
    """Own cbreak cleanup before input-mode entry can be interrupted."""
    try:
        yield keyboard.__enter__()
    finally:
        keyboard.__exit__(None, None, None)


def stop_live(live):
    """Restore the cursor even when Rich only partially acquired the terminal."""
    try:
        live.stop()
    finally:
        # A partial start can hide the cursor before pushing Rich's render
        # hook; stop() can then fail while popping that absent hook.
        live.console.show_cursor(True)


@contextmanager
def live_display(live):
    """Own Rich cleanup before its first refresh, including BaseException."""
    try:
        yield live.__enter__()
    except BaseException:
        try:
            stop_live(live)
        except BaseException:
            pass  # preserve the original interruption after best-effort cleanup
        raise
    else:
        stop_live(live)


@contextmanager
def _blocked_signals(numbers):
    """Keep disposition changes short and restore the caller's exact mask."""
    mask = None
    if hasattr(signal, 'pthread_sigmask'):
        # Read before changing it: interruption after the blocking syscall must
        # not lose the mask we need to restore.
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        if mask is not None:
            signal.pthread_sigmask(signal.SIG_BLOCK, numbers)
        yield mask or set()
    finally:
        if mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            except BaseException:
                # A callback delivered via another thread may interrupt before
                # the syscall. Reapply the saved mask before propagating it.
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                raise


@contextmanager
def termination_signals(*, enabled=True):
    """Let a terminal owner unwind once, then preserve native termination.

    Callers must enable this only while owning a TTY. Batch execution keeps
    native dispositions, including while the main thread is blocked in C. A
    first Python handler restores defaults before raising, so another signal
    can terminate even if cleanup blocks or a callback swallows that exception.
    Python handlers cannot run while native code withholds interpreter signal
    checks: until that first handler runs, TERM/HUP may remain pending. Moving
    cleanup into Python callbacks would have the same latency; a hard native
    hang requires external SIGKILL, without Python or terminal finalization.
    Custom/ignored handlers, blocked signals and worker-thread calls retain
    their existing policy. Nested terminal owners use the outer owner's guard.
    """
    if not enabled or threading.current_thread() is not threading.main_thread():
        yield
        return
    numbers = tuple(getattr(signal, name) for name in ('SIGTERM', 'SIGHUP') if hasattr(signal, name))
    protected = set(numbers) | {signal.SIGINT}
    installed = {}
    received = None
    transitioning = True

    def restore():
        error = None
        for attempt in range(2):
            for number, previous in installed.items():
                try:
                    if signal.getsignal(number) is terminate:
                        signal.signal(number, previous)
                except BaseException as exc:
                    # Finish the others, then retry any still-owned disposition:
                    # interruption can precede or follow the underlying syscall.
                    if error is None:
                        error = exc
        if error is not None:
            raise error

    def terminate(number, frame):
        nonlocal received
        if received is not None:
            signal.signal(number, signal.SIG_DFL)
            signal.raise_signal(number)
            raise SystemExit(128 + number)
        received = number
        if transitioning:
            # Another thread can receive a process signal while this thread has
            # it masked. Defer its Python callback until dispositions are whole.
            return
        with _blocked_signals(protected):
            restore()
        raise TerminationSignal(number)

    try:
        with _blocked_signals(protected) as previous_mask:
            for number in numbers:
                if number not in previous_mask and signal.getsignal(number) == signal.SIG_DFL:
                    installed[number] = signal.SIG_DFL
                    signal.signal(number, terminate)
        transitioning = False
        if received is not None:
            raise TerminationSignal(received)
        yield
    finally:
        transitioning = True
        try:
            with _blocked_signals(protected):
                restore()
        finally:
            if received is not None:
                signal.raise_signal(received)
                raise SystemExit(128 + received)  # a newly installed handler returned
