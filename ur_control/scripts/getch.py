import sys
import termios
import tty
from select import select


def getch(timeout=0.01):
    """
    Retrieves a character from stdin.

    Returns None if no character is available within the timeout.
    Blocks if timeout < 0.
    """
    # If this is being piped to, ignore non-blocking functionality
    if not sys.stdin.isatty():
        return sys.stdin.read(1)
    fileno = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fileno)
    ch = None
    try:
        tty.setraw(fileno)
        rlist = [fileno]
        if timeout >= 0:
            [rlist, _, _] = select(rlist, [], [], timeout)
        if fileno in rlist:
            ch = sys.stdin.read(1)
    except Exception as ex:
        print "getch", ex
        raise OSError
    finally:
        termios.tcsetattr(fileno, termios.TCSADRAIN, old_settings)
    return ch