import subprocess
import sys


def run_cmd(command, log_cmd=False, bash=True):
    sys.stdout.flush()
    sys.stderr.flush()
    if bash:
        if log_cmd:
            print(f"   Running command : bash -c '{command}'")
        subprocess.run(
            ["bash", "-c", command], check=True, stdout=sys.stdout, stderr=subprocess.PIPE
        )
    else:
        raise NotImplementedError("Only bash=True is currently supported for run_cmd")
