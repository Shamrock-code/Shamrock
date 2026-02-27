"""
Test the Phantom dump writer
============================

This example compares a phantom dump with one reproduced by Shamrock.
"""

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


print("-----------------------------------------------------------")
print("----------------   Dump compare utility   -----------------")
print("-----------------------------------------------------------")

# %%
# Load a reference dump
filename = input("Which phantom dump do you want to test ?")
dump_ref = shamrock.load_phantom_dump(filename)

dump_ref.print_state()

# %%
# Start a SPH simulation from the phantom dump
ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_config_from_phantom_dump(dump_ref)
# Set the solver config to be the one stored in cfg
model.set_solver_config(cfg)
# Print the solver config
model.get_current_config().print_status()

model.init_scheduler(int(1e8), 1)

model.init_from_phantom_dump(dump_ref)

dump_2 = model.make_phantom_dump()
dump_2.print_state()

print("-----------------------------------------------------------")
print("-----------------   Comparing the dumps   -----------------")
print("-----------------------------------------------------------")
result_comp = shamrock.compare_phantom_dumps(dump_ref, dump_2)
print(f"Compare phantom dump result : {result_comp}")
print("-----------------------------------------------------------")
print("------------------------   Done   -------------------------")
print("-----------------------------------------------------------")

if not result_comp:
    exit("Dump mismatch reported")
