import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


Npart = 174000
split = int(Npart) * 2


def load_dataset(filename):

    print("Loading", filename)

    dump = shamrock.load_phantom_dump(filename)
    dump.print_state()

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_config_from_phantom_dump(dump)
    # Set the solver config to be the one stored in cfg
    model.set_solver_config(cfg)
    # Print the solver config
    model.get_current_config().print_status()

    model.init_scheduler(split, 1)

    model.init_from_phantom_dump(dump)
    ret = ctx.collect_data()

    del model
    del ctx

    return ret


def L2diff_relat(arr1, pos1, arr2, pos2):
    from scipy.spatial import cKDTree

    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    tree = cKDTree(pos2)
    dists, idxs = tree.query(pos1, k=1)
    matched_arr2 = arr2[idxs]
    return np.sqrt(np.mean((arr1 - matched_arr2) ** 2))

    # Old way without neigh matching
    # return np.sqrt(np.mean((arr1 - arr2) ** 2))


def compare_datasets(istep, dataset1, dataset2):

    if shamrock.sys.world_rank() > 0:
        return

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), dpi=125)

    smarker = 1
    print(len(dataset1["r"]), len(dataset1["rho"]))
    axs[0, 0].scatter(
        dataset1["r"],
        dataset1["rho"],
        s=smarker * 5,
        marker="x",
        c="red",
        rasterized=True,
        label="phantom",
    )
    axs[0, 1].scatter(
        dataset1["r"], dataset1["u"], s=smarker * 5, marker="x", c="red", rasterized=True
    )
    axs[1, 0].scatter(
        dataset1["r"], dataset1["vr"], s=smarker * 5, marker="x", c="red", rasterized=True
    )
    axs[1, 1].scatter(
        dataset1["r"], dataset1["alpha"], s=smarker * 5, marker="x", c="red", rasterized=True
    )

    axs[0, 0].scatter(
        dataset2["r"], dataset2["rho"], s=smarker, c="black", rasterized=True, label="shamrock"
    )
    axs[0, 1].scatter(dataset2["r"], dataset2["u"], s=smarker, c="black", rasterized=True)
    axs[1, 0].scatter(dataset2["r"], dataset2["vr"], s=smarker, c="black", rasterized=True)
    axs[1, 1].scatter(dataset2["r"], dataset2["alpha"], s=smarker, c="black", rasterized=True)

    axs[0, 0].set_ylabel(r"$\rho$")
    axs[1, 0].set_ylabel(r"$v_r$")
    axs[0, 1].set_ylabel(r"$u$")
    axs[1, 1].set_ylabel(r"$\alpha$")

    axs[0, 0].set_xlabel("$r$")
    axs[1, 0].set_xlabel("$r$")
    axs[0, 1].set_xlabel("$r$")
    axs[1, 1].set_xlabel("$r$")

    axs[0, 0].set_xlim(0, 0.5)
    axs[1, 0].set_xlim(0, 0.5)
    axs[0, 1].set_xlim(0, 0.5)
    axs[1, 1].set_xlim(0, 0.5)

    axs[0, 0].legend()

    plt.tight_layout()

    L2r = L2diff_relat(dataset1["r"], dataset1["xyz"], dataset2["r"], dataset2["xyz"])
    L2rho = L2diff_relat(dataset1["rho"], dataset1["xyz"], dataset2["rho"], dataset2["xyz"])
    L2u = L2diff_relat(dataset1["u"], dataset1["xyz"], dataset2["u"], dataset2["xyz"])
    L2vr = L2diff_relat(dataset1["vr"], dataset1["xyz"], dataset2["vr"], dataset2["xyz"])
    L2alpha = L2diff_relat(dataset1["alpha"], dataset1["xyz"], dataset2["alpha"], dataset2["xyz"])

    print("L2r", L2r)
    print("L2rho", L2rho)
    print("L2u", L2u)
    print("L2vr", L2vr)
    print("L2alpha", L2alpha)

    expected_L2 = {
        0: [0.0, 9.009242852618063e-08, 0.0, 0.0, 0.0],
        1: [
            1.849032833754011e-15,
            1.1219057799155968e-07,
            2.999040994476978e-05,
            2.77911044692338e-07,
            1.1107580842674083e-06,
        ],
        10: [
            2.362796968279928e-10,
            1.0893822456258663e-07,
            0.0004010174902848735,
            5.000025464452176e-06,
            0.02366432648382834,
        ],
        100: [
            1.5585397967807125e-08,
            6.155202709399902e-07,
            0.0003118113752459928,
            2.9173459165073988e-05,
            6.432345363293235e-05,
        ],
        1000: [0, 0, 0, 0, 0],
    }

    tols = {
        0: [0, 1e-16, 0, 0, 0],
        1: [1e-20, 1e-16, 1e-16, 1e-18, 1e-20],
        10: [1e-20, 1e-16, 1e-15, 1e-17, 1e-17],
        100: [1e-19, 1e-17, 1e-15, 1e-17, 1e-18],
        1000: [0, 0, 0, 0, 0],
    }

    error = False
    if abs(L2r - expected_L2[istep][0]) > tols[istep][0]:
        error = True
    if abs(L2rho - expected_L2[istep][1]) > tols[istep][1]:
        error = True
    if abs(L2u - expected_L2[istep][2]) > tols[istep][2]:
        error = True
    if abs(L2vr - expected_L2[istep][3]) > tols[istep][3]:
        error = True
    if abs(L2alpha - expected_L2[istep][4]) > tols[istep][4]:
        error = True

    plt.savefig("sedov_blast_phantom_comp_" + str(istep) + ".png")

    if error:
        exit(
            f"Tolerances are not respected, got \n istep={istep}\n"
            + f" got: [{float(L2r)}, {float(L2rho)}, {float(L2u)}, {float(L2vr)}, {float(L2alpha)}] \n"
            + f" expected : [{expected_L2[istep][0]}, {expected_L2[istep][1]}, {expected_L2[istep][2]}, {expected_L2[istep][3]}, {expected_L2[istep][4]}]\n"
            + f" delta : [{(L2r - expected_L2[istep][0])}, {(L2rho - expected_L2[istep][1])}, {(L2u - expected_L2[istep][2])}, {(L2vr - expected_L2[istep][3])}, {(L2alpha - expected_L2[istep][4])}]\n"
            + f" tolerance : [{tols[istep][0]}, {tols[istep][1]}, {tols[istep][2]}, {tols[istep][3]}, {tols[istep][4]}]"
        )


step0000 = load_dataset("reference-files/sedov_blast_phantom/blast_00000")
step0001 = load_dataset("reference-files/sedov_blast_phantom/blast_00001")
step0010 = load_dataset("reference-files/sedov_blast_phantom/blast_00010")
step0100 = load_dataset("reference-files/sedov_blast_phantom/blast_00100")
step1000 = load_dataset("reference-files/sedov_blast_phantom/blast_01000")

print(step0000)


filename_start = "reference-files/sedov_blast_phantom/blast_00000"

dump = shamrock.load_phantom_dump(filename_start)
dump.print_state()


ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_config_from_phantom_dump(dump)
cfg.set_boundary_free()  # try to force some h iterations
# Set the solver config to be the one stored in cfg
model.set_solver_config(cfg)
# Print the solver config
model.get_current_config().print_status()

model.init_scheduler(split, 1)

model.init_from_phantom_dump(dump)

pmass = model.get_particle_mass()


def hpart_to_rho(hpart_array):
    return pmass * (model.get_hfact() / hpart_array) ** 3


def get_testing_sets(dataset):
    ret = {}

    if shamrock.sys.world_rank() > 0:
        return {}

    print("making test dataset, Npart={}".format(len(dataset["xyz"])))

    ret["r"] = np.sqrt(
        dataset["xyz"][:, 0] ** 2 + dataset["xyz"][:, 1] ** 2 + dataset["xyz"][:, 2] ** 2
    )
    ret["rho"] = hpart_to_rho(dataset["hpart"])
    ret["u"] = dataset["uint"]
    ret["vr"] = np.sqrt(
        dataset["vxyz"][:, 0] ** 2 + dataset["vxyz"][:, 1] ** 2 + dataset["vxyz"][:, 2] ** 2
    )
    ret["alpha"] = dataset["alpha_AV"]
    ret["xyz"] = dataset["xyz"]

    # Even though we have neigh matching to compare the datasets
    # We still need the cutoff, hence the sorting + cutoff
    index = np.argsort(ret["r"])

    ret["r"] = ret["r"][index]
    ret["rho"] = ret["rho"][index]
    ret["u"] = ret["u"][index]
    ret["vr"] = ret["vr"][index]
    ret["alpha"] = ret["alpha"][index]
    ret["xyz"] = ret["xyz"][index]

    cutoff = 50000

    ret["r"] = ret["r"][:cutoff]
    ret["rho"] = ret["rho"][:cutoff]
    ret["u"] = ret["u"][:cutoff]
    ret["vr"] = ret["vr"][:cutoff]
    ret["alpha"] = ret["alpha"][:cutoff]
    ret["xyz"] = ret["xyz"][:cutoff]

    return ret


model.evolve_once_override_time(0, 0)

dt = 1e-5
t = 0
for i in range(101):

    if i == 0:
        compare_datasets(i, get_testing_sets(step0000), get_testing_sets(ctx.collect_data()))
    if i == 1:
        compare_datasets(i, get_testing_sets(step0001), get_testing_sets(ctx.collect_data()))
    if i == 10:
        compare_datasets(i, get_testing_sets(step0010), get_testing_sets(ctx.collect_data()))
    if i == 100:
        compare_datasets(i, get_testing_sets(step0100), get_testing_sets(ctx.collect_data()))
    if i == 1000:
        compare_datasets(i, get_testing_sets(step1000), get_testing_sets(ctx.collect_data()))

    model.evolve_once_override_time(0, dt)
    t += dt


# plt.show()
