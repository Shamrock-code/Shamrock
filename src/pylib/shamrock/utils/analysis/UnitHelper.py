def plot_codeu_to_unit(unit_system, name):
    if name is None:
        return "[unitless]", 1

    elif name == "code_unit":
        return "[code unit]", 1

    # Distances
    elif name == "m":
        return "[m]", unit_system.to("m")
    elif name == "cm":
        return "[cm]", unit_system.to("m", pref="c")
    elif name == "km":
        return "[km]", unit_system.to("m", pref="k")
    elif name == "au":
        return "[au]", unit_system.to("au")
    elif name == "pc":
        return "[pc]", unit_system.to("pc")

    # Times
    elif name == "second":
        return "[second]", unit_system.to("s")
    elif name == "minute":
        return "[minute]", unit_system.to("mn")
    elif name == "hour":
        return "[hour]", unit_system.to("hr")
    elif name == "day":
        return "[day]", unit_system.to("dy")
    elif name == "year":
        return "[year]", unit_system.to("yr")
    elif name == "Myr":
        return "[Myr]", unit_system.to("yr", pref="M")
    elif name == "Gyr":
        return "[Gyr]", unit_system.to("yr", pref="G")

    # Inverse times
    elif name == "s^-1":
        return "[s$^{-1}$]", unit_system.to("s", power=-1)
    elif name == "yr^-1":
        return "[yr$^{-1}$]", unit_system.to("yr", power=-1)

    # Surface densities
    elif name == "kg.m^-2":
        return "[$\\mathrm{{kg}} \\cdot \\mathrm{{m}}^{-2}$]", unit_system.to(
            "kg"
        ) * unit_system.to("m", power=-2)
    elif name == "g.cm^-2":
        return "[$\\mathrm{{g}} \\cdot \\mathrm{{cm}}^{-2}$]", unit_system.to(
            "kg", pref="m"
        ) * unit_system.to("m", power=-2, pref="c")

    # Density
    elif name == "kg.m^-3":
        return "[$\\mathrm{{kg}} \\cdot \\mathrm{{m}}^{-3}$]", unit_system.to(
            "kg"
        ) * unit_system.to("m", power=-3)
    elif name == "g.cm^-3":
        return "[$\\mathrm{{g}} \\cdot \\mathrm{{cm}}^{-3}$]", unit_system.to(
            "kg", pref="m"
        ) * unit_system.to("m", power=-3, pref="c")

    # Velocity
    elif name == "m.s^-1":
        return "[$\\mathrm{{m}} \\cdot \\mathrm{{s}}^{-1}$]", unit_system.to("m") * unit_system.to(
            "s", power=-1
        )

    # Acceleration
    elif name == "m.s^-2":
        return "[$\\mathrm{{m}} \\cdot \\mathrm{{s}}^{-2}$]", unit_system.to("m") * unit_system.to(
            "s", power=-2
        )

    else:
        raise ValueError(f"Unknown unit: {name}")
