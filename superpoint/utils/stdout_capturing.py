#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import os
import sys
import subprocess
from threading import Timer
from contextlib import contextmanager

'''
Based on sacred/stdout_capturing.py in project Sacred
https://github.com/IDSIA/sacred
'''


def flush():
    """Try to flush all stdio buffers, both from python and from C."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


# Duplicate stdout and stderr to a file. Inspired by:
# http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
# http://stackoverflow.com/a/651718/1388435
# http://stackoverflow.com/a/22434262/1388435
@contextmanager
def capture_outputs(filename):
    """Duplicate stdout and stderr to a file on the file descriptor level."""
    # with NamedTemporaryFile(mode='w+') as target:
    with open(filename, 'a+') as target:
        original_stdout_fd = 1
        original_stderr_fd = 2
        target_fd = target.fileno()

        # Save a copy of the original stdout and stderr file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        tee_stdout = subprocess.Popen(
            ['tee', '-a', '/dev/stderr'], start_new_session=True,
            stdin=subprocess.PIPE, stderr=target_fd, stdout=1)
        tee_stderr = subprocess.Popen(
            ['tee', '-a', '/dev/stderr'], start_new_session=True,
            stdin=subprocess.PIPE, stderr=target_fd, stdout=2)

        flush()
        os.dup2(tee_stdout.stdin.fileno(), original_stdout_fd)
        os.dup2(tee_stderr.stdin.fileno(), original_stderr_fd)

        try:
            yield
        finally:
            flush()

            # then redirect stdout back to the saved fd
            tee_stdout.stdin.close()
            tee_stderr.stdin.close()

            # restore original fds
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            # wait for completion of the tee processes with timeout
            # implemented using a timer because timeout support is py3 only
            def kill_tees():
                tee_stdout.kill()
                tee_stderr.kill()

            tee_timer = Timer(1, kill_tees)
            try:
                tee_timer.start()
                tee_stdout.wait()
                tee_stderr.wait()
            finally:
                tee_timer.cancel()

            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
