"""
Simple example of unit usage
============================

This simple example shows how to use units in Shamrock
"""

import shamrock

# %%

# The default constructor will provide SI units
si = shamrock.UnitSystem()

# Get the constants in SI
sicte = shamrock.Constants(si)

# %%
print("An au in SI units is", sicte.au())

# %%

# Shamrock unit system is based on the definition of the base units relative to SI ones
# For example to set the time unit one you provide the given time in SI units (seconds).

# Create a unit system with time in years, length in au, mass in solar masses
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)

# Get the physical constants in this unit system
ucte = shamrock.Constants(codeu)
