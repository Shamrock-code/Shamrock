import signal
import sys

from IPython import start_ipython
from traitlets.config.loader import Config

# here the signal interup for sigint is None
# this make ipython freaks out for weird reasons
# registering the handler fix it ...
# i swear python c api is horrible to works with
import shamrock.sys

signal.signal(signal.SIGINT, shamrock.sys.signal_handler)

c = Config()

banner = "SHAMROCK Ipython terminal\n" + "Python %s\n" % sys.version.split("\n")[0]

c.TerminalInteractiveShell.banner1 = banner

c.TerminalInteractiveShell.banner2 = """###
import shamrock
###
"""

start_ipython(config=c)
