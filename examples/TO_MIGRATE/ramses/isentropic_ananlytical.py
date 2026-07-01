
import glob
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# -------------------------------------------------------
# Read all files
# -------------------------------------------------------

# files = sorted(glob.glob("amr_athena_kelvin_helmhotz*.vtk"))
files = sorted(glob.glob("isentropic_vortex_period_*.vtk"))

if len(files) == 0:
    raise RuntimeError("No vtk files found.")


# -------------------------------------------------------
# Compute global limits
# -------------------------------------------------------

rho_min = np.inf
rho_max = -np.inf
meshes = []
for file in files:
 
    mesh = pv.read(file)
    mesh = mesh.compute_cell_sizes()
    meshes.append(mesh)
    rho_min = min(rho_min, mesh["rho"].min())
    rho_max = max(rho_max, mesh["rho"].max())
print("Global rho :", rho_min, rho_max)


def Yee_exact_sol(x, y, t, beta=5.0, gamma=1.4,
              u_inf=1.0, v_inf=0.0,
              Lx=10.0, Ly=10.0,
              x0=5.0, y0=5.0):

    xc = (x0 + u_inf*t) % Lx
    yc = (y0 + v_inf*t) % Ly

    dx = x - xc
    dy = y - yc

    dx -= Lx*np.round(dx/Lx)
    dy -= Ly*np.round(dy/Ly)

    r2 = dx*dx + dy*dy

    T = 1.0 - (gamma-1)*beta**2/(8*gamma*np.pi**2) * np.exp(1-r2)

    rho = T**(1/(gamma-1))

    fac = beta/(2*np.pi) * np.exp((1-r2)/2)

    u = u_inf - dy*fac
    v = v_inf + dx*fac

    p = rho**gamma

    E = p/(gamma-1) + 0.5*rho*(u*u + v*v)

    return rho, rho*u, rho*v, E




def compute_L1_error(mesh, t, L=10.0,base=32,sz=2):

    centers = mesh.cell_centers().points

    # Convert to physical coordinates
    dx_phys = L/(base*sz)

    x = centers[:,0] * dx_phys
    y = centers[:,1] * dx_phys

    rho_num = mesh["rho"]

    rho_exact = Yee_exact_sol(x, y, t)[0]

    L1 = np.mean(np.abs(rho_num-rho_exact))

    return L1





def extract_mesh_datas(meshes, L=10., sz=2, base=32):

    extent = [0, L, 0, L]
    _2d_datas = {"xu":[], "yu":[], "rho2d":[]}
    _1d_datas = {"xu":[], "rho1d":[]}
    for i, mesh in enumerate(meshes):
        centers = mesh.cell_centers().points

        dx_phys = L / (sz * base)

        x = centers[:,0] * dx_phys
        y = centers[:,1] * dx_phys
        z = centers[:,2] * dx_phys

        rho = mesh["rho"]

        z_c = L/2.
        y_c = L/2.

        dx = np.cbrt(mesh["Volume"])
        print(f"dx = {dx}\n")
        tol = 0.51*dx.max()


        ## 2d datas
        mask = np.abs(z-z_c)<tol
        x_plane = x[mask]
        y_plane = y[mask]
        rho_plane = rho[mask]

        xu = np.unique(x_plane)
        yu = np.unique(y_plane)
        nx = len(xu)
        ny = len(yu)

        rho_xy = np.full((ny,nx), np.nan)

        for xx,yy,rr in zip(x_plane, y_plane,rho_plane):
            ix = np.argmin(np.abs(xu-xx))
            iy = np.argmin(np.abs(yu-yy))

            rho_xy[iy,ix] = rr

        plt.figure(figsize=(6,6))
        
        im = plt.imshow(
            rho_xy,
            origin="lower",
            extent=extent,
            cmap="rainbow",
            vmin=rho_min,
            vmax=rho_max
        )


        # ----------------------------------------------------
        # Contour lines
        # ----------------------------------------------------
        levels = [ 0.65, 0.75,0.85, 0.95]

        X, Y = np.meshgrid(xu, yu, indexing="xy")

        cs = plt.contour(
            X,
            Y,
            rho_xy,
            levels=levels,
            colors="white",     
            linewidths=1.1,
        )

        plt.clabel(
            cs,
            inline=True,
            fmt="%.2f",
            fontsize=8,
        )

        cbar = plt.colorbar(im,   shrink=0.8 )
        cbar.set_label(r"$\rho$")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect("equal")
 
        
        _2d_datas["xu"].append(xu)
        _2d_datas["yu"].append(yu)
        _2d_datas["rho2d"].append(rho_xy)


        # ## 1d datas

        y_plane = np.unique(y[np.abs(z-z_c) < tol])

        iy = np.argmin(np.abs(y_plane - y_c))
        y_slice = y_plane[iy]

        # mask_1d = (
        #     np.abs(z-z_c) < tol
        # ) & (
        #     np.abs(y-y_slice) < 1e-12
        # )

        z_plane = np.unique(z)
        iz = np.argmin(np.abs(z_plane - z_c))
        z_slice = z_plane[iz]

        mask_1d = (
            np.abs(z - z_slice) < 1e-12
        ) & (
            np.abs(y - y_slice) < 1e-12
        )
        print("Number of points:", np.count_nonzero(mask_1d))
        print(f"y_slice = {y_slice}\n")
        print(f"z_slice = {z_slice}\n")


        x_line = x[mask_1d]
        rho_line = rho[mask_1d]  
        idx = np.argsort(x_line)
        x_line = x_line[idx]
        rho_line = rho_line[idx]
        print(f"x_line = \t {x_line}\n\n")
        print(f"rho_line = \t {rho_line}\n\n")
        plt.title(f"t = {i*L}")

        plt.savefig(f"isentropic_vortex_at_t_{i*L}.pdf")

        _1d_datas["xu"].append(x_line)
        _1d_datas["rho1d"].append(rho_line)


    return _1d_datas,_2d_datas



_1d_datas,_2d_datas = extract_mesh_datas(meshes,L=10.)



nb_profile = len(_1d_datas["xu"])
X = _1d_datas["xu"][0]
L = 10.
Yc = L/2
extent = [0, L, 0, L]

# -------------------------------------------------------
# L1 error versus time
# -------------------------------------------------------



times = []
L1_errors = []

for i, mesh in enumerate(meshes):

    t = i * L          # 0,10,20,...

    L1 = compute_L1_error(
        mesh,
        t=t,
        L=L,
        base=32,
        sz=2,
    )

    times.append(t)
    L1_errors.append(L1)

    print(f"t = {t:5.1f}   L1 = {L1:.8e}")


_1d_exact_rho = rho_exact = np.array([
    Yee_exact_sol(xx,Yc,t=0.0)[0]
    for xx in X
])






##-----------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(24,7),
    dpi=300,
    # gridspec_kw={"width_ratios":[1.2,1.0,1.25]},
    # constrained_layout=True
)
fig.subplots_adjust(
    wspace=0.35,
    left=0.06,
    right=0.97,
    bottom=0.12,
    top=0.93,
)


X_plot = X 


axes[0].plot(X_plot, _1d_exact_rho, "-", lw=2, label="t = 0")

for i in range(1, nb_profile):
    axes[0].plot(
        X_plot,
        _1d_datas["rho1d"][i],
        "o-",
        ms=2,
        label=f"t = {i*L}"
    )


axes[0].set_xlabel("x")
axes[0].set_ylabel(r"$\rho$")
axes[0].legend(
    loc="lower left",
    fontsize=14,
)
axes[0].grid(True)

#----
axes[1].plot(
    times,
    L1_errors,
    "o-",
    lw=2,
    ms=5,
)

axes[1].set_yscale("log")
axes[1].set_xlabel("Time")
axes[1].set_ylabel(r"$L_1$")
axes[1].grid(True)

#-----
im = axes[2].imshow(
            _2d_datas["rho2d"][-1],
            origin="lower",
            extent=extent,
            cmap="rainbow",
            vmin=rho_min,
            vmax=rho_max
        )

levels = [ 0.65, 0.75,0.85, 0.95]

_X, _Y = np.meshgrid(_2d_datas["xu"][-1], _2d_datas["yu"][-1], indexing="xy")

cs = axes[2].contour(
    _X,
    _Y,
    _2d_datas["rho2d"][-1],
    levels=levels,
    colors="white",     
    linewidths=1.1,
)

axes[2].clabel(
    cs,
    inline=True,
    fmt="%.2f",
    fontsize=8,
)

cbar = fig.colorbar(
    im,
    ax=axes[2],
    shrink=0.8,
    pad=0.03,
)
cbar.set_label(r"$\rho$")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_title(f" t = {(nb_profile-1)*L}")
axes[2].set_aspect("equal")


plt.savefig(f"1d_isentropic_vortex_in_time_2.pdf")



# plt.figure(figsize=(7,4))
# plt.plot(X, _1d_exact_rho,"-",
#     lw=2,
#     label="Exact")

# for i in range(1,nb_profile):
#     plt.plot(X, _1d_datas["rho1d"][i], "o-", ms=2,label=f"Numerical t = {i*L}")

# plt.xlabel("x")
# plt.ylabel(r"$\rho$")
# plt.legend()
# plt.grid(True)
# plt.savefig(f"1d_isentropic_vortex_at_t_.pdf")


# plt.figure(figsize=(6,4))

# plt.plot(
#     times,
#     L1_errors,
#     "o-",
#     lw=2,
#     ms=5,
# )

# plt.xlabel("Time")
# plt.ylabel(r"$L_1(\rho)$")

# plt.grid(True)
# plt.tight_layout()

# plt.savefig("L1_error_vs_time.pdf")
