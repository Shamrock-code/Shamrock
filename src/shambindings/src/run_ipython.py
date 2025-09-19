import signal
import sys

from IPython import start_ipython
from traitlets.config.loader import Config

c = Config()

banner = "SHAMROCK Ipython terminal\n" + "Python %s\n" % sys.version.split("\n")[0]

c.TerminalInteractiveShell.banner1 = banner

c.TerminalInteractiveShell.banner2 = """###
import shamrock
###
"""

start_ipython(config=c)
