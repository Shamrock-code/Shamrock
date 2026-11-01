"""
CI test: unit system API checks
===============================

This example checks how ``shamrock.UnitSystem`` and ``shamrock.Constants``
behave when used from Python: the unit and prefix names, the meaning of
``get`` / ``to`` with ``power`` and ``pref``, the physical constants, and the
JSON round trip.

It runs as part of the documentation build, so any change to that behaviour
makes the build fail.
"""

import math

import shamrock

# %%
# Helpers
# ^^^^^^^
# Every check compares floats with a relative tolerance.


def check(actual, expected, what, rel_tol=1e-12):
    assert math.isclose(actual, expected, rel_tol=rel_tol), (
        f"{what}: got {actual!r}, expected {expected!r}"
    )


def check_raises(exc_type, func, what):
    try:
        func()
    except exc_type:
        return
    raise AssertionError(f"{what}: expected {exc_type.__name__} to be raised")


# %%
# Unit and prefix names
# ^^^^^^^^^^^^^^^^^^^^^
# Every unit has a long and a short name, and both give the same value.
# Pairs are listed as ``(long name, short name)``.

unit_names = [
    ("second", "s"),
    ("metre", "m"),
    ("kilogram", "kg"),
    ("Ampere", "A"),
    ("Kelvin", "K"),
    ("mole", "mol"),
    ("candela", "cd"),
    ("Hertz", "Hz"),
    ("Newton", "N"),
    ("Pascal", "Pa"),
    ("Joule", "J"),
    ("Watt", "W"),
    ("Coulomb", "C"),
    ("Volt", "V"),
    ("Farad", "F"),
    ("Ohm", "ohm"),
    ("Siemens", "S"),
    ("Weber", "Wb"),
    ("Tesla", "T"),
    ("Henry", "H"),
    ("lumens", "lm"),
    ("lux", "lx"),
    ("Becquerel", "Bq"),
    ("Gray", "Gy"),
    ("Sievert", "Sv"),
    ("katal", "kat"),
    ("minutes", "mn"),
    ("hours", "hr"),
    ("days", "dy"),
    ("years", "yr"),
    ("astronomical_unit", "au"),
    ("light_year", "ly"),
    ("parsec", "pc"),
    ("solar_radius", "rsol"),
    ("earth_radius", "rearth"),
    ("solar_mass", "sol_mass"),
    ("electron_volt", "eV"),
    ("ergs", "erg"),
    ("british_pint", "pint"),
]

# Prefixes as ``(long name, short name, power of ten)``.
# Note that ``"m"`` as a prefix is milli, not metre.
prefix_names = [
    ("tera", "T", 12),
    ("giga", "G", 9),
    ("mega", "M", 6),
    ("kilo", "k", 3),
    ("hecto", "hect", 2),
    ("deca", "dec", 1),
    ("None", "_", 0),
    ("centi", "c", -2),
    ("milli", "m", -3),
    ("micro", "mu", -6),
    ("nano", "n", -9),
    ("pico", "p", -12),
    ("femto", "f", -15),
]

# %%
# SI unit system
# ^^^^^^^^^^^^^^
# The default constructor gives SI units, so every base unit is 1.

si = shamrock.UnitSystem()

check(si.get("m"), 1.0, 'si.get("m")')
check(si.get("metre"), 1.0, 'si.get("metre")')
check(si.get("s"), 1.0, 'si.get("s")')
check(si.get("kg"), 1.0, 'si.get("kg")')
check(si.get("s", power=-1), 1.0, 'si.get("s", power=-1)')

# %%
# ``pref`` takes the long or the short prefix name, positionally or by keyword.
check(si.get("m", 1, "kilo"), 1e3, 'si.get("m", 1, "kilo")')
check(si.get("m", pref="k"), 1e3, 'si.get("m", pref="k")')
check(si.get("m", pref="c"), 1e-2, 'si.get("m", pref="c")')
check(si.get("kg", pref="m"), 1e-3, 'si.get("kg", pref="m")')

for long_name, short_name, exponent in prefix_names:
    check(si.get("m", pref=long_name), 10.0**exponent, f'si.get("m", pref="{long_name}")')
    check(si.get("m", pref=short_name), 10.0**exponent, f'si.get("m", pref="{short_name}")')

# %%
# ``power`` raises the prefixed unit, so the prefix is raised too.
check(si.get("m", power=2, pref="k"), 1e6, 'si.get("m", power=2, pref="k")')
check(si.get("m", power=-3, pref="c"), 1e6, 'si.get("m", power=-3, pref="c")')

# %%
# ``get`` gives the value of one unit in the system's units, ``to`` gives the
# value of one system unit in the given unit. ``to`` is the inverse of ``get``.
check(si.get("au"), 149597870700.0, 'si.get("au")')
check(si.to("au"), 1.0 / 149597870700.0, 'si.to("au")')
check(si.get("pc"), 3.0857e16, 'si.get("pc")')
check(si.get("yr"), 31557600.0, 'si.get("yr")')
check(si.to("m", pref="k"), 1e-3, 'si.to("m", pref="k")')

# %%
# Conversions used by ``shamrock.utils.analysis.UnitHelper``

# Distances
check(si.to("m"), 1.0, 'si.to("m")')
check(si.to("m", pref="c"), 1e2, 'si.to("m", pref="c")')
check(si.to("m", pref="k"), 1e-3, 'si.to("m", pref="k")')
check(si.to("au"), 1.0 / 149597870700.0, 'si.to("au")')
check(si.to("pc"), 1.0 / 3.0857e16, 'si.to("pc")')

# Times
check(si.to("s"), 1.0, 'si.to("s")')
check(si.to("mn"), 1.0 / 60.0, 'si.to("mn")')
check(si.to("hr"), 1.0 / 3600.0, 'si.to("hr")')
check(si.to("dy"), 1.0 / 86400.0, 'si.to("dy")')
check(si.to("yr"), 1.0 / 31557600.0, 'si.to("yr")')

# Inverse times
check(si.to("s", power=-1), 1.0, 'si.to("s", power=-1)')
check(si.to("yr", power=-1), 31557600.0, 'si.to("yr", power=-1)')

# Masses, surface densities and densities
check(si.to("kg"), 1.0, 'si.to("kg")')
check(si.to("kg", pref="m"), 1e3, 'si.to("kg", pref="m")')
check(si.to("m", power=-2), 1.0, 'si.to("m", power=-2)')
check(si.to("m", power=-2, pref="c"), 1e-4, 'si.to("m", power=-2, pref="c")')
check(si.to("m", power=-3), 1.0, 'si.to("m", power=-3)')
check(si.to("m", power=-3, pref="c"), 1e-6, 'si.to("m", power=-3, pref="c")')

# Accelerations and magnetic field
check(si.to("s", power=-2), 1.0, 'si.to("s", power=-2)')
check(si.to("T"), 1.0, 'si.to("T")')

# %%
# Physical constants in SI
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Each constant takes an optional ``power``.

sicte = shamrock.Constants(si)

si_constants = {
    "au": 149597870700.0,
    "G": 6.6743015e-11,
    "c": 299792458.0,
    "year": 31557600.0,
    "sol_mass": 1.98847e30,
    "second": 1.0,
    "kb": 1.380649e-23,
    "dalton": 1.66053906892e-27,
    "mu_0": 1.2566370621219e-6,
}

for name, value in si_constants.items():
    func = getattr(sicte, name)
    check(func(), value, f"sicte.{name}()")
    check(func(power=1), value, f"sicte.{name}(power=1)")
    check(func(power=2), value**2, f"sicte.{name}(power=2)")
    check(func(power=-1), 1.0 / value, f"sicte.{name}(power=-1)")

# %%
# Every constant exposed to Python, by name.
constant_names = [
    "delta_nu_cs",
    "c",
    "h",
    "e",
    "k",
    "Na",
    "Kcd",
    "second",
    "minute",
    "hour",
    "day",
    "year",
    "au",
    "astronomical_unit",
    "light_year",
    "parsec",
    "solar_radius",
    "earth_radius",
    "planck_length",
    "proton_mass",
    "electron_mass",
    "earth_mass",
    "jupiter_mass",
    "sol_mass",
    "planck_mass",
    "dalton",
    "guiness_density",
    "G",
    "hbar",
    "mu_0",
    "Z_0",
    "epsilon_0",
    "ke",
    "kb",
    "sigma",
]

for name in constant_names:
    value = getattr(sicte, name)()
    assert math.isfinite(value) and value > 0, f"sicte.{name}() = {value!r}"
    check(getattr(sicte, name)(power=2), value**2, f"sicte.{name}(power=2)")

# %%
# Astrophysical unit system
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Time in Myr, length in au and mass in solar masses, built from literals.

unit_time = 3.15576e13
unit_length = 1.495978707e11
unit_mass = 1.98847e30

astro = shamrock.UnitSystem(unit_time=unit_time, unit_length=unit_length, unit_mass=unit_mass)

check(astro.to("au"), 1.0, 'astro.to("au")')
check(astro.get("au"), 1.0, 'astro.get("au")')
check(astro.to("sol_mass"), 1.0, 'astro.to("sol_mass")')
check(astro.to("yr"), 1e6, 'astro.to("yr")')
check(astro.to("s"), unit_time, 'astro.to("s")')
check(astro.to("m"), unit_length, 'astro.to("m")')
check(astro.to("kg"), unit_mass, 'astro.to("kg")')
check(astro.get("s"), 1.0 / unit_time, 'astro.get("s")')
check(astro.get("T"), unit_time**2 / unit_mass, 'astro.get("T")')
check(astro.to("m", power=-3, pref="c"), unit_length**-3 * 1e-6, 'astro.to("m", -3, "c")')

# %%
# Long and short unit names resolve to the same unit.
for long_name, short_name in unit_names:
    check(astro.get(long_name), astro.get(short_name), f'astro.get("{long_name}")')
    check(astro.to(long_name), astro.to(short_name), f'astro.to("{long_name}")')

# %%
# A prefix applies once to the whole unit, even for a unit built from other
# units: ``Myr`` is ``1e6 yr`` and ``Gyr`` is ``1e9 yr``.
check(si.to("yr", pref="M"), 1e-6 / 31557600.0, 'si.to("yr", pref="M")')
check(astro.to("yr", pref="M"), 1.0, 'astro.to("yr", pref="M")')
check(astro.to("yr", pref="G"), 1e-3, 'astro.to("yr", pref="G")')

# %%
# Constants in the astrophysical system
astrocte = shamrock.Constants(astro)

check(astrocte.au(), 1.0, "astrocte.au()")
check(astrocte.sol_mass(), 1.0, "astrocte.sol_mass()")
check(astrocte.year(), 1e-6, "astrocte.year()")
check(astrocte.second(), 1.0 / unit_time, "astrocte.second()")
check(astrocte.c(), 299792458.0 * unit_time / unit_length, "astrocte.c()")
check(
    astrocte.G(),
    6.6743015e-11 * unit_mass * unit_time**2 / unit_length**3,
    "astrocte.G()",
)
check(astrocte.G(power=-1), 1.0 / astrocte.G(), "astrocte.G(power=-1)")

# In au, solar masses and years, G is close to 4 pi^2 (Kepler's third law)
check(astrocte.G() / 1e12, 4 * math.pi**2, "astrocte.G() vs 4 pi^2", rel_tol=1e-3)

print("G in au^3 Msun^-1 Myr^-2 :", astrocte.G())

# %%
# JSON round trip
# ^^^^^^^^^^^^^^^
# ``to_json`` returns a dict with the seven base units in SI, and
# ``from_json`` sets a unit system from such a dict.

astro_json = astro.to_json()
print(astro_json)

expected_json = {
    "unit_time": unit_time,
    "unit_length": unit_length,
    "unit_mass": unit_mass,
    "unit_current": 1.0,
    "unit_temperature": 1.0,
    "unit_qte": 1.0,
    "unit_lumint": 1.0,
}
assert set(astro_json.keys()) == set(expected_json.keys()), astro_json
for key, value in expected_json.items():
    check(astro_json[key], value, f"astro.to_json()[{key!r}]")

reloaded = shamrock.UnitSystem()
reloaded.from_json(astro_json)

for key, value in expected_json.items():
    check(reloaded.to_json()[key], value, f"reloaded.to_json()[{key!r}]")
for long_name, short_name in unit_names:
    check(reloaded.get(short_name), astro.get(short_name), f'reloaded.get("{short_name}")')
check(shamrock.Constants(reloaded).G(), astrocte.G(), "Constants(reloaded).G()")

# %%
# Errors
# ^^^^^^
# Unknown unit or prefix names raise ``ValueError``, and the constructor
# only takes keyword arguments.

check_raises(ValueError, lambda: si.get("not_a_unit"), 'si.get("not_a_unit")')
check_raises(ValueError, lambda: si.to("not_a_unit"), 'si.to("not_a_unit")')
check_raises(ValueError, lambda: si.get("m", pref="not_a_prefix"), 'pref="not_a_prefix"')
check_raises(TypeError, lambda: shamrock.UnitSystem(1.0), "UnitSystem(1.0)")

print("All unit system API checks passed")
