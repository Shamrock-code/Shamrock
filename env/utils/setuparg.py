class SetupArg:
    """argument that will be passed to the machine setups"""

    def __init__(self, argv, builddir, shamrockdir, buildtype, pylib, lib_mode):
        self.argv = argv
        self.builddir = builddir
        self.shamrockdir = shamrockdir
        self.buildtype = buildtype
        self.pylib = pylib
        self.lib_mode = lib_mode
