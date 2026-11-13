"""Tell interactive users when a newer SO3LR release exists on GitHub.

At most one request per day, in a background thread with a short timeout; the
answer is cached under the user cache directory and shown on a later command.
Nothing runs in batch/scheduler/CI jobs, with --quiet, without a terminal, or
when SO3LR_NO_UPDATE_CHECK=1. The request carries no user information.
"""
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

from .terminal import SCHEDULER_VARIABLES

RELEASES_URL = 'https://api.github.com/repos/general-molecular-simulations/so3lr/releases/latest'
UPGRADE_HINT = 'pip install -U git+https://github.com/general-molecular-simulations/so3lr.git'
CHECK_EVERY = 24 * 3600     # seconds between network checks
TIMEOUT = 2.0               # seconds for one request
_BATCH_VARIABLES = (*SCHEDULER_VARIABLES, 'CI')


def cache_path(environ=None):
    environ = os.environ if environ is None else environ
    root = environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return Path(root) / 'so3lr' / 'latest-release.json'


def allowed(environ=None, stdin=None, stderr=None, quiet=False):
    """Only interactive, non-scheduler sessions that did not opt out."""
    environ = os.environ if environ is None else environ
    stdin = sys.stdin if stdin is None else stdin
    stderr = sys.stderr if stderr is None else stderr
    if quiet or environ.get('SO3LR_NO_UPDATE_CHECK', '').strip() not in ('', '0'):
        return False
    if any(environ.get(name) for name in _BATCH_VARIABLES) or environ.get('TERM') == 'dumb':
        return False
    return all(bool(getattr(stream, 'isatty', lambda: False)()) for stream in (stdin, stderr))


def _version_key(text):
    """'v0.10.2' -> (0, 10, 2); None when the tag is not a plain release version."""
    match = re.fullmatch(r'v?(\d+(?:\.\d+)*)', str(text).strip())
    return tuple(int(part) for part in match.group(1).split('.')) if match else None


def newer(latest, current):
    latest_key, current_key = _version_key(latest), _version_key(current)
    return latest_key is not None and current_key is not None and latest_key > current_key


def _read(path):
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _fetch(url=RELEASES_URL, timeout=TIMEOUT):
    from urllib.request import Request, urlopen
    request = Request(url, headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'so3lr-cli'})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https URL)
        return str(json.load(response).get('tag_name') or '')


def refresh(path=None, now=None, fetch=_fetch):
    """Update the cache when it is older than a day; never raises."""
    path = cache_path() if path is None else Path(path)
    now = time.time() if now is None else now
    cached = _read(path)
    checked = cached.get('checked_at')
    if isinstance(checked, (int, float)) and now - checked < CHECK_EVERY:
        return
    tag = cached.get('latest', '')
    # Claim the day before the request: a short command can exit while it is in
    # flight (the thread is a daemon), and must not make every next command ask again.
    if not _write(path, now, tag):
        return
    try:
        tag = fetch()
    except Exception:
        return  # offline or rate-limited: try again tomorrow
    _write(path, now, tag)


def _write(path, now, tag):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'checked_at': now, 'latest': tag}))
        return True
    except OSError:
        return False


def start_refresh(path=None):
    """Refresh the cache in a daemon thread so no command waits for the network."""
    thread = threading.Thread(target=refresh, kwargs={'path': path}, name='so3lr-update-check', daemon=True)
    thread.start()
    return thread


def notice(current, path=None):
    """The one-line notice for a newer cached release, or ''."""
    latest = _read(cache_path() if path is None else path).get('latest', '')
    if not newer(latest, current):
        return ''
    return f'SO3LR {str(latest).lstrip("v")} is available (you have {current}). Upgrade: {UPGRADE_HINT}'
