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
ESCAPE_TIMEOUT = 0.12


class Back(click.Abort):
    """Esc or Backspace in a menu: return to the previous step.

    A subclass of Abort, so callers that do not handle it still cancel safely.
    """


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
    if buffer.startswith('\x1bO'):
        if len(buffer) >= 3:
            return 'ignore', buffer[3:]
        # A timed-out SS3 prefix is one unsupported key, not Escape then O.
        # O is a viewer shortcut and must not leak out of an incomplete arrow.
        return ('ignore', '') if final else (None, buffer)
    char, rest = buffer[0], buffer[1:]
    if char in ('\x1b', '\x7f', '\x08'):  # Escape, Backspace (DEL or ^H)
        return (char if literal else 'back'), rest
    if char in ('\x03', '\x04', '\x1a'):
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
                if self.ready(ESCAPE_TIMEOUT):
                    self.buffer += self.read()
                    continue
                event, remainder = decode_key(self.buffer, final=True)
            self.buffer = remainder
            if event == 'cancel':
                raise click.Abort()
            if event == 'back':
                raise Back()
            if event != 'ignore':
                return event


@contextmanager
def terminal_keys():
    """Restore terminal input mode on selection, errors and cancellation.

    The caller owns the termination-signal guard (see ``choose``).
    """
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
    from . import theme
    hint = theme.controls('↑/↓: move · Enter: choose · Esc: back', single_line=True)
    lines = [Text(terminal_text(title), style=theme.STRONG, no_wrap=True, overflow='ellipsis'), hint]
    if start:
        lines.append(Text(f'  ↑ {start} more', style=theme.MUTED))
    for index in range(start, end):
        line = Text(no_wrap=True, overflow='ellipsis')
        if index == selected:
            line.append('› ', style=theme.ACCENT_BOLD)
            line.append(terminal_text(choices[index]), style=theme.ACCENT_BOLD)
        else:
            line.append('  ' + terminal_text(choices[index]))
        lines.append(line)
    if end < len(choices):
        lines.append(Text(f'  ↓ {len(choices) - end} more', style=theme.MUTED))
    return Group(*lines)


def choose(console, title, choices, default=0):
    """Return the chosen index. Esc/Backspace raise Back; Ctrl+C cancels."""
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
    from . import theme
    # Answered questions recede; the next question keeps the focus.
    from .formatting import ascii_only_console
    ascii_only = ascii_only_console(console)
    # Grey, not green: green is kept for a run that met its goal.
    summary = Text(terminal_text(theme.status_glyph('completed') + ' ', ascii_only), style=theme.FAINT)
    summary.append(terminal_text(f'{title}: {choices[selected]}', ascii_only), style=theme.MUTED)
    console.print(summary)
    return selected
