"""
Shamrock DeviceBuffer usage
===========================

This example shows how to use the shamrock.backends.DeviceBuffer_f64 class
for GPU-accelerated data storage and manipulation.
"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Create a new DeviceBuffer for f64 (double precision floats)
buffer = shamrock.backends.DeviceBuffer_f64()

print(f"Initial buffer size: {buffer.get_size()}")

# %%
# Resize the buffer to hold 10 elements
buffer.resize(10)
print(f"Buffer size after resize: {buffer.get_size()}")

# %%
# Set individual values using set_val_at_idx
for i in range(10):
    buffer.set_val_at_idx(i, float(i * 2.5))

print("Values set individually:")
for i in range(buffer.get_size()):
    print(f"buffer[{i}] = {buffer.get_val_at_idx(i)}")

# %%
# Copy data from a Python list/numpy array
data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
buffer.copy_from_stdvec(data)

print(f"\nAfter copying from list: {data}")
for i in range(buffer.get_size()):
    print(f"buffer[{i}] = {buffer.get_val_at_idx(i)}")

# %%
# Copy data back to a Python list
result = buffer.copy_to_stdvec()
print(f"\nCopied back to Python list: {result}")

# %%
# Working with numpy arrays
np_data = np.linspace(0.0, 1.0, 20)
print(f"Original numpy array (size {len(np_data)}): {np_data}")

# Resize buffer to match numpy array size
buffer.resize(len(np_data))
buffer.copy_from_stdvec(np_data.tolist())

print(f"Buffer size after numpy copy: {buffer.get_size()}")

# %%
# Perform some operations: multiply each element by 2
for i in range(buffer.get_size()):
    current_val = buffer.get_val_at_idx(i)
    buffer.set_val_at_idx(i, current_val * 2.0)

# Copy back and convert to numpy array
modified_data = np.array(buffer.copy_to_stdvec())
print(f"Modified data (multiplied by 2): {modified_data}")

# %%
# Demonstrate with larger dataset
rng = np.random.default_rng(42)  # Use modern random generator with seed for reproducibility
large_data = rng.random(1000)
buffer.resize(len(large_data))
buffer.copy_from_stdvec(large_data.tolist())

print(f"Processed large dataset of size: {buffer.get_size()}")
print(f"First 10 elements: {buffer.copy_to_stdvec()[:10]}")
print(f"Last 10 elements: {buffer.copy_to_stdvec()[-10:]}")
