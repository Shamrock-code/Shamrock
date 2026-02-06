"""
Phantom related utilities.
"""

import os

import shamrock.sys


def parse_in_file(in_file):
    """
    Parse a Phantom .in file and return a dictionary of the parameters.
    """
    with open(in_file, "r") as f:
        lines = f.readlines()

    params = {}

    for line in lines:
        # Skip empty lines and comment lines
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        # Check if line contains an equals sign
        if "=" in line:
            # Split by '=' to get variable name and value part
            parts = line.split("=", 1)
            var_name = parts[0].strip()

            # Get value part (everything after =)
            value_part = parts[1]

            # Remove comment if present (text after !)
            if "!" in value_part:
                value_part = value_part.split("!")[0]

            # Strip whitespace from value
            value = value_part.strip()

            # Try to convert to appropriate type
            # Check for boolean
            if value == "T":
                value = True
            elif value == "F":
                value = False
            else:
                # Try to convert to number
                try:
                    # Try integer first
                    if "." not in value and "E" not in value and "e" not in value:
                        value = int(value)
                    else:
                        # Try float
                        value = float(value)
                except ValueError:
                    # Keep as string if conversion fails
                    pass

            params[var_name] = value

    return params


def load_simulation(simulation_path, dump_file_name=None, in_file_name=None, do_print=True):
    """
    Load a Phantom simulation into a Shamrock model.
    """

    if do_print and shamrock.sys.world_rank() == 0:
        print("-----------------------------------------------------------")
        print("----------------   Phantom dump loading   -----------------")
        print("-----------------------------------------------------------")

    if in_file_name is not None:
        in_file_path = os.path.join(simulation_path, in_file_name)
        in_params = parse_in_file(in_file_path)
    else:
        in_params = None

    if dump_file_name is None:
        if in_file_name is not None:
            dump_file_name = in_params["dumpfile"]

        else:
            raise ValueError("Either dump_file_name or in_file_name must be provided")

    dump_path = os.path.join(simulation_path, dump_file_name)

    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Loading phantom dump from: ", dump_path)

    # setup = dump finish with .tmp
    is_setup_file = dump_file_name.endswith(".tmp")

    # Open the phantom dump
    dump = shamrock.load_phantom_dump(dump_path)

    # Start a SPH simulation from the phantom dump
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Generating Shamrock solver config from phantom dump")
    cfg = model.gen_config_from_phantom_dump(dump)
    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Setting Shamrock solver config")
    # Set the solver config to be the one stored in cfg
    model.set_solver_config(cfg)

    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Initializing domain scheduler")

    model.init_scheduler(int(1e8), 1)

    if do_print and shamrock.sys.world_rank() == 0:
        print(f" - Initializing from phantom dump (setup file: {is_setup_file})")

    if is_setup_file:
        model.init_from_phantom_dump(dump, 0.05)
    else:
        model.init_from_phantom_dump(dump, 1.0)

    # Print infos
    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Shamrock solver config:")
        model.get_current_config().print_status()

        if in_params is not None:
            print(" - Phantom input file parameters:")
            for key, value in in_params.items():
                print(f"{key}: {value}")

        # print("Dump state:")
        # dump.print_state()

    return ctx, model, in_params


def run_phantom_simulation(simulation_folder, sim_name, callback=None):
    """
    Run a Phantom simulation in Shamrock.
    """

    input_file_name = sim_name + ".in"

    ctx, model, in_params = shamrock.utils.phantom.load_simulation(
        simulation_folder, in_file_name=input_file_name
    )

    dump_file_name = in_params["dumpfile"]

    # phantom dumps are 00000.tmp if before start and then sim_name_{:05d}
    # parse the dump number
    dump_number = int(dump_file_name.split("_")[1].split(".")[0])
    print(f"Dump number: {dump_number}")

    dtmax = float(in_params["dtmax"])
    tmax = float(in_params["tmax"])

    def get_ph_dump_file_name(dump_number):
        return f"{sim_name}_{dump_number:05d}"

    def get_ph_dump_name(dump_number):
        return os.path.join(simulation_folder, get_ph_dump_file_name(dump_number))

    def do_dump(dump_number):
        if shamrock.sys.world_rank() == 0:
            print("-----------------------------------------------------------")
            print("----------------   Phantom dump saving   -----------------")
            print("-----------------------------------------------------------")
            print(f" - Saving dump {dump_number} to {get_ph_dump_name(dump_number)}")
        dump = model.make_phantom_dump()
        dump.save_dump(get_ph_dump_name(dump_number))

        # replace dumpfile in the input file
        lines = []
        with open(os.path.join(simulation_folder, f"{sim_name}.in"), "r") as f:
            lines = f.readlines()
        with open(os.path.join(simulation_folder, f"{sim_name}.in"), "w") as f:
            for line in lines:
                if "dumpfile" in line:
                    line = f"            dumpfile = {get_ph_dump_file_name(dump_number)}  ! dump file to start from\n"
                f.write(line)

        # if .tmp is there remove it
        if (
            os.path.exists(os.path.join(simulation_folder, f"{sim_name}_00000.tmp"))
            and shamrock.sys.world_rank() == 0
        ):
            os.remove(os.path.join(simulation_folder, f"{sim_name}_00000.tmp"))

        if callback is not None:
            callback(ctx, model, dump_number)

    model.set_next_dt(0)
    model.timestep()

    do_dump(dump_number)

    dump_number += 1

    # evolve until tmax in increments of dtmax
    last_step = False
    while not last_step:
        next_time = model.get_time() + dtmax

        if next_time > tmax:
            next_time = tmax
            last_step = True

        model.evolve_until(next_time)

        do_dump(dump_number)
        dump_number += 1

    return ctx, model, in_params
