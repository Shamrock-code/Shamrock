"""
Open a phantom dump
============================

This simple example shows how to open a phantom dump in shamrock
"""

# %%
# Download a phantom dump
dump_folder = "_to_trash"
import os

os.system("mkdir -p " + dump_folder)

url = "https://raw.githubusercontent.com/Shamrock-code/reference-files/refs/heads/main/blast_00010"

filename = dump_folder + "/blast_00010"

from urllib.request import urlretrieve

urlretrieve(url, filename)

# %%
# Open the phantom dump

import shamrock

dump = shamrock.load_phantom_dump(filename)

# %%
# Print the data

dump.print_state()
