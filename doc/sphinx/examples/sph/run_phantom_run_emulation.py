"""
Perform a Phantom run in Shamrock
==========================================

Setup from a phantom dump and run according to the input file
"""

import os
import shutil
from urllib.request import urlretrieve

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

dump_folder = "_to_trash/phantom_test_sim"
if shamrock.sys.world_rank() == 0:
    # remove the folder if it exists (ok if it does not exist)
    if os.path.exists(dump_folder):
        shutil.rmtree(dump_folder)
    os.makedirs(dump_folder, exist_ok=True)

shamrock.sys.mpi_barrier()

input_file_name = "disc.in"
dump_file_name = "disc_00000.tmp"

input_file_path = os.path.join(dump_folder, input_file_name)
dump_file_path = os.path.join(dump_folder, dump_file_name)

input_file_url = "https://raw.githubusercontent.com/Shamrock-code/reference-files/refs/heads/main/phantom_disc_simulation/disc.in"
dump_file_url = "https://raw.githubusercontent.com/Shamrock-code/reference-files/refs/heads/main/phantom_disc_simulation/disc_00000.tmp"

shamrock.utils.url.download_file(input_file_url, input_file_path)
shamrock.utils.url.download_file(dump_file_url, dump_file_path)

shamrock.utils.phantom.run_phantom_simulation(dump_folder, "disc")
