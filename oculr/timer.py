from functools import wraps
from timeit import default_timer


timer = default_timer


def print_with_timestamp(start_time: float, *args):
    elapsed_sec = timer() - start_time
    if elapsed_sec < 1:
        time_format = '5.3f'
    elif elapsed_sec < 10:
        time_format = '5.2f'
    elif elapsed_sec < 1000:
        time_format = '5.1f'
    else:
        time_format = '5.0f'
    print(f'[{elapsed_sec:{time_format}}]', *args)


def timed(f):
    """Wrap function with the timer that returns tuple: result, elapsed_time."""
    @wraps(f)
    def _wrap(*args, **kw):
        start = timer()
        result = f(*args, **kw)
        elapsed = timer() - start
        return result, elapsed
    return _wrap
