"""Small keyboard-driven selections without another terminal dependency."""
from contextlib import contextmanager
import os
import re
import sys

import click

from .formatting import terminal_text
from .terminal import live_display, termination_signals


_KEYS = {
    '\x1b[A': 'up', '\x1bOA': 'up', '\xe0H': 'up', '\x00H': 'up',
    '\x1b[B': 'down', '\x1bOB': 'down', '\xe0P': 'down', '\x00P': 'down',
    '\x1b[C': 'right', '\x1bOC': 'right', '\xe0M': 'right', '\x00M': 'right',
    '\x1b[D': 'left', '\x1bOD': 'left', '\xe0K': 'left', '\x00K': 'left',
    '\x1b[H': 'home', '\x1bOH': 'home', '\x1b[1~': 'home', '\xe0G': 'home',
    '\x1b[F': 'end', '\x1bOF': 'end', '\x1b[4~': 'end', '\xe0O': 'end',
    '\x1b[5~': 'page_up', '\xe0I': 'page_up',
    '\x1b[6~': 'page_down', '\xe0Q': 'page_down',
}
_CSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def decode_key(buffer, *, final=False, literal=False):
    """Consume one key, retaining incomplete escape sequences for the next read.

    ``final`` is set after a short escape timeout, distinguishing Escape from
    an arrow key even when the terminal sends that key in several chunks.
    ``literal`` preserves ordinary characters for monitor shortcuts; prompt
    callers retain their existing enter/cancel/ignore behavior.
    """
    if not buffer:
        return None, buffer
    for sequence, name in _KEYS.items():
        if buffer.startswith(sequence):
            return name, buffer[len(sequence):]
    if not final and any(sequence.startswith(buffer) for sequence in _KEYS):
        return None, buffer
    # Consume unsupported escape sequences as a unit: their payload must not
    # accidentally become a shortcut (notably the O in application-cursor keys).
    if buffer.startswith('\x1b['):
        match = _CSI.match(buffer)
        if match:
            return 'ignore', buffer[match.end():]
        if not final and len(buffer) < 64:
            return None, buffer
        return 'ignore', buffer[min(len(buffer), 64):]
    if buffer.startswith('\x1bO') and len(buffer) >= 3:
        return 'ignore', buffer[3:]
    char, rest = buffer[0], buffer[1:]
    if char in ('\x03', '\x04', '\x1a', '\x1b'):
        return (char if literal else 'cancel'), rest
    if char in ('\r', '\n'):
        return (char if literal else 'enter'), rest if literal else rest.lstrip('\r\n')
    return (char if literal else 'ignore'), rest


class _Keys:
    def __init__(self, read, ready):
        self.read = read
        self.ready = ready
        self.buffer = ''

    def next(self):
        while True:
            if not self.buffer:
                self.buffer = self.read()
                if not self.buffer:
                    raise click.Abort()
            event, remainder = decode_key(self.buffer)
            if event is None:
                # Most keyboards send an entire escape sequence together; SSH
                # or a busy terminal can split it. Keep character input active while
                # waiting so canonical line buffering cannot swallow the tail.
                if self.ready(0.12):
                    self.buffer += self.read()
                    continue
                event, remainder = decode_key(self.buffer, final=True)
            self.buffer = remainder
            if event == 'cancel':
                raise click.Abort()
            if event != 'ignore':
                return event


@contextmanager
def terminal_keys():
    """Restore terminal input mode on selection, errors and cancellation."""
    with termination_signals(enabled=bool(getattr(sys.stdin, 'isatty', lambda: False)())):
        with _terminal_keys() as keys:
            yield keys


@contextmanager
def _terminal_keys():
    if os.name == 'nt':
        import msvcrt
        import time

        def ready(timeout):
            until = time.monotonic() + timeout
            while time.monotonic() < until:
                if msvcrt.kbhit():
                    return True
                time.sleep(0.005)
            return False

        yield _Keys(lambda: click.getchar(echo=False), ready)
        return
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        # Cbreak preserves output newline processing while disabling canonical
        # input and echo. Raw mode also disables ONLCR and breaks multiline
        # Rich rendering on terminals where stdin and stdout share a device.
        # Keep typeahead for the next prompt; neither flush pending input nor
        # read beyond the key this selection is currently consuming.
        tty.setcbreak(fd, termios.TCSANOW)
        yield _Keys(lambda: os.read(fd, 1).decode('utf-8', errors='replace'),
                    lambda timeout: bool(select.select([fd], [], [], timeout)[0]))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def arrow_input_available(console):
    return (console.is_terminal and bool(getattr(sys.stdin, 'isatty', lambda: False)())
            and os.environ.get('TERM') != 'dumb')


def selection_view(title, choices, selected=0, height=24):
    """The native selection view, also usable for previews without key input."""
    from rich.console import Group
    from rich.text import Text
    if not choices:
        raise ValueError('A selection needs at least one choice.')
    selected = max(0, min(selected, len(choices) - 1))
    visible = max(3, min(14, height - 6))
    start = max(0, min(selected - visible // 2, len(choices) - visible))
    end = min(len(choices), start + visible)
    lines = [Text(terminal_text(title), style='bold', no_wrap=True, overflow='ellipsis'),
             Text('↑/↓ move · Enter choose · Esc cancel', style='dim')]
    if start:
        lines.append(Text(f'  ↑ {start} more', style='dim'))
    for index in range(start, end):
        marker = '›' if index == selected else ' '
        lines.append(Text(f'{marker} {terminal_text(choices[index])}',
                          style='bold yellow' if index == selected else '',
                          no_wrap=True, overflow='ellipsis'))
    if end < len(choices):
        lines.append(Text(f'  ↓ {len(choices) - end} more', style='dim'))
    return Group(*lines)


def choose(console, title, choices, default=0):
    """Return the chosen index. Escape/Ctrl+C cancels the enclosing wizard."""
    from rich.live import Live
    from rich.text import Text

    if not choices:
        raise ValueError('A selection needs at least one choice.')
    selected = max(0, min(default, len(choices) - 1))
    if not arrow_input_available(console):
        # CliRunner and nonterminal test streams cannot provide raw key events.
        # This fallback also remains useful for simpler terminal integrations.
        console.print(Text(terminal_text(title), style='bold'))
        for number, choice in enumerate(choices, 1):
            console.print(Text(f'  {number}. {terminal_text(choice)}'))
        return click.prompt('Choose', type=click.IntRange(1, len(choices)), default=selected + 1) - 1

    visible = max(3, min(14, console.height - 6))

    def render():
        return selection_view(title, choices, selected, height=console.height)

    with termination_signals(enabled=arrow_input_available(console)), live_display(Live(render(), console=console, transient=True, auto_refresh=False,
              redirect_stdout=False, redirect_stderr=False)) as live:
        with terminal_keys() as keys:
            while True:
                event = keys.next()
                if event == 'enter':
                    break
                if event == 'up':
                    selected = (selected - 1) % len(choices)
                elif event == 'down':
                    selected = (selected + 1) % len(choices)
                elif event == 'home':
                    selected = 0
                elif event == 'end':
                    selected = len(choices) - 1
                elif event == 'page_up':
                    selected = max(0, selected - visible)
                elif event == 'page_down':
                    selected = min(len(choices) - 1, selected + visible)
                live.update(render(), refresh=True)
    console.print(Text(terminal_text(f'{title}: {choices[selected]}')))
    return selected
