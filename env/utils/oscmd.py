
import subprocess

def run_cmd(command, log_cmd = False, bash=True):

    if bash:
        if(log_cmd):
            print(f"   Running command : bash -c '{command}'")
        subprocess.run(["bash", "-c", command], check=True)
    else:
        raise "Unimplemented"
