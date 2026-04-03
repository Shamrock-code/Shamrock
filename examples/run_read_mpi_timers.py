"""
Read Shamrock MPI timers
===========================

"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
before_mpi_timers = shamrock.comm.get_timers()
print(f"before_mpi_timers = {before_mpi_timers}")

# %%
# Force some MPI time in the most rude way possible
for i in range(1000):
    shamrock.sys.mpi_barrier()

# %%
after_mpi_timers = shamrock.comm.get_timers()
print(f"after_mpi_timers = {after_mpi_timers}")

# %%
# Compute the delta
delta = shamrock.comm.mpi_timers_delta(before_mpi_timers, after_mpi_timers)

keys = [k for k in delta.keys()]
keys.sort()
for k in keys:
    print(f"{k} = {delta[k]} s")
