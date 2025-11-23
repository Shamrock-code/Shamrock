"""
FMM math demo
=============

This example shows how to use the FMM maths to compute the force between two points
"""

# %%
# As always, we start by importing the necessary libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import shamrock

# %%
# Utilities
# ^^^^^^^^^
# You can ignore this first block, it just contains some utility functions to draw the AABB and the arrows
# We only defines the function `draw_aabb` and `draw_arrow`, which are used to draw the AABB and the arrows in the plots
# and the function `draw_box_pair`, which is used to draw the box pair with all the vectors needed to compute the FMM force

# %%
# .. raw:: html
#
#   <details>
#   <summary><a>Click here to expand the utility code</a></summary>
#


def draw_aabb(ax, aabb, color, alpha):
    """
    Draw a 3D AABB in matplotlib

    Parameters
    ----------
    ax : matplotlib.Axes3D
        The axis to draw the AABB on
    aabb : shamrock.math.AABB_f64_3
        The AABB to draw
    color : str
        The color of the AABB
    alpha : float
        The transparency of the AABB
    """
    xmin, ymin, zmin = aabb.lower
    xmax, ymax, zmax = aabb.upper

    points = [
        aabb.lower,
        (aabb.lower[0], aabb.lower[1], aabb.upper[2]),
        (aabb.lower[0], aabb.upper[1], aabb.lower[2]),
        (aabb.lower[0], aabb.upper[1], aabb.upper[2]),
        (aabb.upper[0], aabb.lower[1], aabb.lower[2]),
        (aabb.upper[0], aabb.lower[1], aabb.upper[2]),
        (aabb.upper[0], aabb.upper[1], aabb.lower[2]),
        aabb.upper,
    ]

    faces = [
        [points[0], points[1], points[3], points[2]],
        [points[4], points[5], points[7], points[6]],
        [points[0], points[1], points[5], points[4]],
        [points[2], points[3], points[7], points[6]],
        [points[0], points[2], points[6], points[4]],
        [points[1], points[3], points[7], points[5]],
    ]

    edges = [
        [points[0], points[1]],
        [points[0], points[2]],
        [points[0], points[4]],
        [points[1], points[3]],
        [points[1], points[5]],
        [points[2], points[3]],
        [points[2], points[6]],
        [points[3], points[7]],
        [points[4], points[5]],
        [points[4], points[6]],
        [points[5], points[7]],
        [points[6], points[7]],
    ]

    collection = Poly3DCollection(faces, alpha=alpha, color=color)
    ax.add_collection3d(collection)

    edge_collection = Line3DCollection(edges, color="k", alpha=alpha)
    ax.add_collection3d(edge_collection)


def draw_arrow(ax, p1, p2, color, label, arr_scale=0.1):
    length = np.linalg.norm(np.array(p2) - np.array(p1))
    arrow_length_ratio = arr_scale / length
    ax.quiver(
        p1[0],
        p1[1],
        p1[2],
        p2[0] - p1[0],
        p2[1] - p1[1],
        p2[2] - p1[2],
        color=color,
        label=label,
        arrow_length_ratio=arrow_length_ratio,
    )


# %%
# .. raw:: html
#
#   </details>


# %%
#
# FMM force computation
# ^^^^^^^^^^^^^^^^^^^^^
#
# Let's start by assuming that we have two particles at positions :math:`\mathbf{x}_i` and
# :math:`\mathbf{x}_j` contained in two boxes (:math:`A` and :math:`B`) whose centers are at positions
# :math:`\mathbf{s}_a` and :math:`\mathbf{s}_b` respectively.
# The positions of the particles relative to their respective boxes are then:
#
# .. math::
#    \mathbf{a}_i = \mathbf{x}_i - \mathbf{s}_a \\
#    \mathbf{b}_j = \mathbf{x}_j - \mathbf{s}_b
#
# and the distance between the centers of the boxes is:
#
# .. math::
#    \mathbf{r} = \mathbf{s}_b - \mathbf{s}_a
#
# This implies that the distance between the two particles is:
#
# .. math::
#    \mathbf{x}_j - \mathbf{x}_i = \mathbf{r} + \mathbf{b}_j - \mathbf{a}_i
#
# If we denote the Green function for an inverse distance :math:`G(\mathbf{x}) = 1 / \vert\vert\mathbf{x}\vert\vert`, then the potential exerted onto particle :math:`i` is:
#
# .. math::
#    \Phi_i = \Phi (\mathbf{x}_i) &= \int  - \frac{\mathcal{G} \rho(\mathbf{x}_j)}{\vert\vert\mathbf{x}_i - \mathbf{x}_j\vert\vert} d\mathbf{x}_j \\
#    &= - \mathcal{G} \int \rho(\mathbf{x}_j) G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j
#
# and the force exerted onto particle :math:`i` is:
#
# .. math::
#    \mathbf{f}_i = -\nabla_i \Phi (\mathbf{x}_i) &= \int - \nabla_i \frac{\mathcal{G} \rho(\mathbf{x}_j)}{\vert\vert\mathbf{x}_i - \mathbf{x}_j\vert\vert} d\mathbf{x}_j \\
#    &= \mathcal{G} \int \rho(\mathbf{x}_j) \nabla_i G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j \\
#    &= -\mathcal{G} \int \rho(\mathbf{x}_j) \nabla_j G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j
#
# Now let's expand the green function in a Taylor series to order :math:`p`.
#
# .. math::
#    G(\mathbf{x}_j - \mathbf{x}_i) &= G(\mathbf{r} + \mathbf{b}_j - \mathbf{a}_i) \\
#       &= \sum_{k = 0}^p \frac{(-1)^k}{k!} \mathbf{a}_i^{(k)} \cdot \sum_{n=0}^{p-k} \frac{1}{n!} D_{n+k} \cdot \mathbf{b}_j^{(n)}
#
# where :math:`D_{n} = \nabla^{(n)}_r G(\mathbf{r})` is the n-th order derivative of the Green
# function and the operator :math:`\mathbf{a}_i^{(k)}` is the tensor product of :math:`k` :math:`\mathbf{a}_i` vectors.

# %%
#
# Similarly the gradient of the green function is:
#
# .. math::
#    \nabla_j G(\mathbf{x}_j - \mathbf{x}_i) &= \nabla_r G(\mathbf{r} + \mathbf{b}_j - \mathbf{a}_i) \\
#       &= \sum_{k = 0}^p \frac{(-1)^k}{k!} \mathbf{a}_i^{(k)} \cdot \sum_{n=0}^{p-k} \frac{1}{n!} D_{n+k+1} \cdot \mathbf{b}_j^{(n)}
#
# Now we can plug that back into the expression for the force & potential:
#
# .. math::
#    \Phi_i &= - \mathcal{G} \int \rho(\mathbf{x}_j) G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j \\
#    &= - \mathcal{G} \sum_{k = 0}^p \frac{1}{k!} \mathbf{a}_i^{(k)} \cdot \underbrace{(-1)^k \sum_{n=0}^{p-k} \frac{1}{n!} D_{n+k} \cdot \underbrace{\int \rho(\mathbf{x}_j) \mathbf{b}_j^{(n)} d\mathbf{x}_j}_{Q_n^B}}_{M_{p,k}} \\
#
# .. math::
#    \mathbf{f}_i &= -\mathcal{G} \int \rho(\mathbf{x}_j) \nabla_j G(\mathbf{x}_j - \mathbf{x}_i) d\mathbf{x}_j \\
#    &= -\mathcal{G}  \sum_{k = 0}^p \frac{1}{k!} \mathbf{a}_i^{(k)} \cdot \underbrace{(-1)^k \sum_{n=0}^{p-k} \frac{1}{n!} D_{n+k+1} \cdot  \underbrace{\int \rho(\mathbf{x}_j)\mathbf{b}_j^{(n)} d\mathbf{x}_j}_{Q_n^B}}_{dM_{p,k} = M_{p+1,k+1}}
#

# %%
#
# As one can tell sadly the two expressions while similar do not share the same terms.
#
# I will not go in this rabit hole of using the same expansion for both now but the idea is to
# use the primitive of the force which is the same expansion as the force but with the primitive
# of :math:`\mathbf{a}_i^{(k)}` instead.
#
# .. math::
#    \Phi_i  = - \int \mathbf{f}_i =  -\mathcal{G}  \sum_{k = 0}^p \frac{1}{k!} \int\mathbf{a}_i^{(k)} \cdot {M_{p+1,k+1}}
#

# %%
#
# Mass moments
# ^^^^^^^^^^^^

# %%
# .. raw:: html
#
#   <details>
#   <summary><a>def plot_mass_moment_case(s_B,box_B_size,x_j):</a></summary>
#


def plot_mass_moment_case(s_B, box_B_size, x_j):
    box_B = shamrock.math.AABB_f64_3(
        (
            s_B[0] - box_B_size / 2,
            s_B[1] - box_B_size / 2,
            s_B[2] - box_B_size / 2,
        ),
        (
            s_B[0] + box_B_size / 2,
            s_B[1] + box_B_size / 2,
            s_B[2] + box_B_size / 2,
        ),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_arrow(ax, s_B, x_j, "black", "$b_j = x_j - s_B$")

    ax.scatter(s_B[0], s_B[1], s_B[2], color="black", label="s_B")

    ax.scatter(x_j[0], x_j[1], x_j[2], color="red", label="$x_j$")

    draw_aabb(ax, box_B, "blue", 0.2)

    center_view = (0.0, 0.0, 0.0)
    view_size = 2.0
    ax.set_xlim(center_view[0] - view_size / 2, center_view[0] + view_size / 2)
    ax.set_ylim(center_view[1] - view_size / 2, center_view[1] + view_size / 2)
    ax.set_zlim(center_view[2] - view_size / 2, center_view[2] + view_size / 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax


# %%
# .. raw:: html
#
#   </details>

# %%
# Let's start with the following

s_B = (0, 0, 0)
box_B_size = 1

x_j = (0.2, 0.2, 0.2)
m_j = 1

b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])

ax = plot_mass_moment_case(s_B, box_B_size, x_j)
plt.title("Mass moment illustration")
plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
plt.show()


# %%
# Here the mass moment of a set of particles (here only one) of mass :math:`m_j` is
#
# .. math::
#    {Q_n^B} &= \int \rho(\mathbf{x}_j) \mathbf{b}_j^{(n)} d\mathbf{x}_j\\
#            &= \sum_j m_j \mathbf{b}_j^{(n)}
#
# In Shamrock python bindings the function

# %%
# .. code-block::
#
#        shamrock.math.SymTensorCollection_f64_<low order>_<high order>.from_vec(b_j)

# %%
# will return the collection of symetrical tensors :math:`\mathbf{b}_j^{(n)}` for n in between `<low order>` and `<high order>`
# Here are the values of the tensors :math:`{Q_n^B}` from order 0 up to 5 using shamrock symmetrical tensor collections

Q_n_B = shamrock.math.SymTensorCollection_f64_0_5.from_vec(b_j)
Q_n_B *= m_j

print("Q_n_B =", Q_n_B)

# %%
# Now if we take a displacment that is only along the x axis we get null components in the Q_n_B if for cases that do not only exhibit x

s_B = (0, 0, 0)
box_B_size = 1

x_j = (0.5, 0.0, 0.0)
m_j = 1

b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])

ax = plot_mass_moment_case(s_B, box_B_size, x_j)
plt.title("Mass moment illustration")
plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
plt.show()

Q_n_B = shamrock.math.SymTensorCollection_f64_0_5.from_vec(b_j)
Q_n_B *= m_j

print("Q_n_B =", Q_n_B)


# %%
#
# Gravitational moments
# ^^^^^^^^^^^^^^^^^^^^^


# %%
# .. raw:: html
#
#   <details>
#   <summary><a>def plot_mass_moment_case(s_B,box_B_size,x_j):</a></summary>
#
def plot_grav_moment_case(s_A, box_A_size, s_B, box_B_size, x_j):
    box_A = shamrock.math.AABB_f64_3(
        (
            s_A[0] - box_A_size / 2,
            s_A[1] - box_A_size / 2,
            s_A[2] - box_A_size / 2,
        ),
        (
            s_A[0] + box_A_size / 2,
            s_A[1] + box_A_size / 2,
            s_A[2] + box_A_size / 2,
        ),
    )

    box_B = shamrock.math.AABB_f64_3(
        (
            s_B[0] - box_B_size / 2,
            s_B[1] - box_B_size / 2,
            s_B[2] - box_B_size / 2,
        ),
        (
            s_B[0] + box_B_size / 2,
            s_B[1] + box_B_size / 2,
            s_B[2] + box_B_size / 2,
        ),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_arrow(ax, s_B, x_j, "black", "$b_j = x_j - s_B$")

    draw_arrow(ax, s_A, s_B, "purple", "$r = s_B - s_A$")

    ax.scatter(s_A[0], s_A[1], s_A[2], color="black", label="s_A")

    ax.scatter(s_B[0], s_B[1], s_B[2], color="green", label="s_B")

    ax.scatter(x_j[0], x_j[1], x_j[2], color="red", label="$x_j$")

    draw_aabb(ax, box_A, "blue", 0.1)
    draw_aabb(ax, box_B, "red", 0.1)

    center_view = (0.5, 0.0, 0.0)
    view_size = 2.0
    ax.set_xlim(center_view[0] - view_size / 2, center_view[0] + view_size / 2)
    ax.set_ylim(center_view[1] - view_size / 2, center_view[1] + view_size / 2)
    ax.set_zlim(center_view[2] - view_size / 2, center_view[2] + view_size / 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax


# %%
# .. raw:: html
#
#   </details>

# %%
# Let's now show the example of a gravitational moment, for the following case
s_B = (0, 0, 0)
s_A = (1, 0, 0)

box_B_size = 0.5
box_A_size = 0.5

x_j = (0.2, 0.2, 0.0)
m_j = 1

b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])
r = (s_B[0] - s_A[0], s_B[1] - s_A[1], s_B[2] - s_A[2])

ax = plot_grav_moment_case(s_A, box_A_size, s_B, box_B_size, x_j)
plt.title("Grav moment illustration")
plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
plt.show()

# %%
# The mass moment :math:`{Q_n^B}` is
Q_n_B = shamrock.math.SymTensorCollection_f64_0_4.from_vec(b_j)
Q_n_B *= m_j
print("Q_n_B =", Q_n_B)

# %%
# The green function n'th gradients :math:`D_{n+k+1}` are
D_n = shamrock.phys.green_func_grav_cartesian_1_5(r)
print("D_n =", D_n)

# %%
# And finally the gravitational moments :math:`dM_{p,k}` are
dM_k = shamrock.phys.get_dM_mat_5(D_n, Q_n_B)
print("dM_k =", dM_k)

# %%
#
# From Gravitational moments to force
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# %%
# .. raw:: html
#
#   <details>
#   <summary><a>def plot_fmm_case(s_A,box_A_size,x_i,s_B,box_B_size,x_j, f_i_fmm, f_i_exact):</a></summary>
#
def plot_fmm_case(s_A, box_A_size, x_i, s_B, box_B_size, x_j, f_i_fmm, f_i_exact, fscale_fact):
    box_A = shamrock.math.AABB_f64_3(
        (
            s_A[0] - box_A_size / 2,
            s_A[1] - box_A_size / 2,
            s_A[2] - box_A_size / 2,
        ),
        (
            s_A[0] + box_A_size / 2,
            s_A[1] + box_A_size / 2,
            s_A[2] + box_A_size / 2,
        ),
    )

    box_B = shamrock.math.AABB_f64_3(
        (
            s_B[0] - box_B_size / 2,
            s_B[1] - box_B_size / 2,
            s_B[2] - box_B_size / 2,
        ),
        (
            s_B[0] + box_B_size / 2,
            s_B[1] + box_B_size / 2,
            s_B[2] + box_B_size / 2,
        ),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_arrow(ax, s_B, x_j, "black", "$b_j = x_j - s_B$")
    draw_arrow(ax, s_A, x_i, "blue", "$a_i = x_i - s_A$")

    draw_arrow(ax, s_A, s_B, "purple", "$r = s_B - s_A$")

    ax.scatter(s_A[0], s_A[1], s_A[2], color="black", label="s_A")

    ax.scatter(s_B[0], s_B[1], s_B[2], color="green", label="s_B")

    ax.scatter(x_i[0], x_i[1], x_i[2], color="orange", label="$x_i$")

    ax.scatter(x_j[0], x_j[1], x_j[2], color="red", label="$x_j$")

    draw_arrow(ax, x_i, x_i + force_i * fscale_fact, "green", "$f_i$")
    draw_arrow(ax, x_i, x_i + force_i_exact * fscale_fact, "red", "$f_i$ (exact)")

    abs_error = np.linalg.norm(force_i - force_i_exact)
    rel_error = abs_error / np.linalg.norm(force_i_exact)

    draw_aabb(ax, box_A, "blue", 0.1)
    draw_aabb(ax, box_B, "red", 0.1)

    center_view = (0.5, 0.0, 0.0)
    view_size = 2.0
    ax.set_xlim(center_view[0] - view_size / 2, center_view[0] + view_size / 2)
    ax.set_ylim(center_view[1] - view_size / 2, center_view[1] + view_size / 2)
    ax.set_zlim(center_view[2] - view_size / 2, center_view[2] + view_size / 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax, rel_error, abs_error


# %%
# .. raw:: html
#
#   </details>


# %%
# Now let's put everything together to get a FMM force
# We start with the following parameters (see figure below for the representation)

s_B = (0, 0, 0)
s_A = (1, 0, 0)

box_B_size = 0.5
box_A_size = 0.5

x_j = (0.2, 0.2, 0.0)
x_i = (1.2, 0.2, 0.0)
m_j = 1
m_i = 1

b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])
r = (s_B[0] - s_A[0], s_B[1] - s_A[1], s_B[2] - s_A[2])
a_i = (x_i[0] - s_A[0], x_i[1] - s_A[1], x_i[2] - s_A[2])

# %%
Q_n_B = shamrock.math.SymTensorCollection_f64_0_4.from_vec(b_j)
Q_n_B *= m_j
print("Q_n_B =", Q_n_B)

# %%
D_n = shamrock.phys.green_func_grav_cartesian_1_5(r)
print("D_n =", D_n)

# %%
dM_k = shamrock.phys.get_dM_mat_5(D_n, Q_n_B)
print("dM_k =", dM_k)

# %%
a_k = shamrock.math.SymTensorCollection_f64_0_4.from_vec(a_i)
print("a_k =", a_k)

# %%
result = shamrock.phys.contract_grav_moment_to_force_5(a_k, dM_k)
Gconst = 1  # let's just set the grav constant to 1
force_i = -Gconst * np.array(result)
print("force_i =", force_i)


# %%
# Now we just need the analytical force to compare
def analytic_force_i(x_i, x_j, Gconst):
    force_i_direct = (x_j[0] - x_i[0], x_j[1] - x_i[1], x_j[2] - x_i[2])
    force_i_direct /= np.linalg.norm(force_i_direct) ** 3
    force_i_direct *= m_i
    return force_i_direct


force_i_exact = analytic_force_i(x_i, x_j, Gconst)
print("force_i_exact =", force_i_exact)

# %%
# This yields the following case
ax, rel_error, abs_error = plot_fmm_case(
    s_A, box_A_size, x_i, s_B, box_B_size, x_j, force_i, force_i_exact, 0.5
)

plt.title(f"FMM, rel error={rel_error}")
plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
plt.show()

print("force_i =", force_i)
print("force_i_exact =", force_i_exact)
print("abs error =", abs_error)
print("rel error =", rel_error)

# %%
# And yeah the error is insanelly low, but it is the special case where :math:`a_i = b_j`.
# Anyway now let's wrap all of that mess into a function that does it all and see how the error
# changes depending on the configure and order of the expansion.

# %%
# FMM in box
# ^^^^^^^^^^
# The following is the function to do the same as above but for whatever order


def run_fmm(x_i, x_j, s_A, s_B, m_j, order, do_print):

    b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])
    r = (s_B[0] - s_A[0], s_B[1] - s_A[1], s_B[2] - s_A[2])
    a_i = (x_i[0] - s_A[0], x_i[1] - s_A[1], x_i[2] - s_A[2])

    if do_print:
        print("x_i =", x_i)
        print("x_j =", x_j)
        print("s_A =", s_A)
        print("s_B =", s_B)
        print("b_j =", b_j)
        print("r =", r)
        print("a_i =", a_i)

    # compute the tensor product of the displacment
    if order == 1:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_0.from_vec(b_j)
    elif order == 2:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_1.from_vec(b_j)
    elif order == 3:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_2.from_vec(b_j)
    elif order == 4:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_3.from_vec(b_j)
    elif order == 5:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_4.from_vec(b_j)
    else:
        raise ValueError("Invalid order")

    # multiply by mass to get the mass moment
    Q_n_B *= m_j

    if do_print:
        print("Q_n_B =", Q_n_B)

    # green function gradients
    if order == 1:
        D_n = shamrock.phys.green_func_grav_cartesian_1_1(r)
    elif order == 2:
        D_n = shamrock.phys.green_func_grav_cartesian_1_2(r)
    elif order == 3:
        D_n = shamrock.phys.green_func_grav_cartesian_1_3(r)
    elif order == 4:
        D_n = shamrock.phys.green_func_grav_cartesian_1_4(r)
    elif order == 5:
        D_n = shamrock.phys.green_func_grav_cartesian_1_5(r)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("D_n =", D_n)

    if order == 1:
        dM_k = shamrock.phys.get_dM_mat_1(D_n, Q_n_B)
    elif order == 2:
        dM_k = shamrock.phys.get_dM_mat_2(D_n, Q_n_B)
    elif order == 3:
        dM_k = shamrock.phys.get_dM_mat_3(D_n, Q_n_B)
    elif order == 4:
        dM_k = shamrock.phys.get_dM_mat_4(D_n, Q_n_B)
    elif order == 5:
        dM_k = shamrock.phys.get_dM_mat_5(D_n, Q_n_B)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("dM_k =", dM_k)

    if order == 1:
        a_k = shamrock.math.SymTensorCollection_f64_0_0.from_vec(a_i)
    elif order == 2:
        a_k = shamrock.math.SymTensorCollection_f64_0_1.from_vec(a_i)
    elif order == 3:
        a_k = shamrock.math.SymTensorCollection_f64_0_2.from_vec(a_i)
    elif order == 4:
        a_k = shamrock.math.SymTensorCollection_f64_0_3.from_vec(a_i)
    elif order == 5:
        a_k = shamrock.math.SymTensorCollection_f64_0_4.from_vec(a_i)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("a_k =", a_k)

    if order == 1:
        result = shamrock.phys.contract_grav_moment_to_force_1(a_k, dM_k)
    elif order == 2:
        result = shamrock.phys.contract_grav_moment_to_force_2(a_k, dM_k)
    elif order == 3:
        result = shamrock.phys.contract_grav_moment_to_force_3(a_k, dM_k)
    elif order == 4:
        result = shamrock.phys.contract_grav_moment_to_force_4(a_k, dM_k)
    elif order == 5:
        result = shamrock.phys.contract_grav_moment_to_force_5(a_k, dM_k)
    else:
        raise ValueError("Invalid order")

    Gconst = 1  # let's just set the grav constant to 1
    force_i = -Gconst * np.array(result)
    if do_print:
        print("force_i =", force_i)

    force_i_exact = analytic_force_i(x_i, x_j, Gconst)
    if do_print:
        print("force_i_exact =", force_i_exact)

    b_A_size = np.linalg.norm(np.array(s_A) - np.array(x_i))
    b_B_size = np.linalg.norm(np.array(s_B) - np.array(x_j))
    b_dist = np.linalg.norm(np.array(s_A) - np.array(s_B))

    angle = (b_A_size + b_B_size) / b_dist

    if do_print:
        print("b_A_size =", b_A_size)
        print("b_B_size =", b_B_size)
        print("b_dist =", b_dist)
        print("angle =", angle)

    return force_i, force_i_exact, angle


# %%
# Let's try with some new parameters
s_B = (0, 0, 0)
s_A = (1, 0, 0)

box_B_size = 0.5
box_A_size = 0.5

x_j = (0.2, 0.2, 0.0)
x_i = (1.2, 0.2, 0.2)
m_j = 1
m_i = 1

force_i, force_i_exact, angle = run_fmm(x_i, x_j, s_A, s_B, m_j, order=5, do_print=True)
ax, rel_error, abs_error = plot_fmm_case(
    s_A, box_A_size, x_i, s_B, box_B_size, x_j, force_i, force_i_exact, 0.5
)

plt.title(f"FMM angle={angle:.5f} rel error={rel_error:.2e}")
plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
plt.show()

print("force_i =", force_i)
print("force_i_exact =", force_i_exact)
print("abs error =", abs_error)
print("rel error =", rel_error)

# %%
# Varying the order of the expansion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# sphinx_gallery_multi_image = "single"

s_B = (0, 0, 0)
s_A = (1, 0, 0)

box_B_size = 0.5
box_A_size = 0.5

x_j = (0.2, 0.2, 0.0)
x_i = (0.8, 0.2, 0.2)
m_j = 1
m_i = 1


for order in range(1, 6):
    print("--------------------------------")
    print(f"Running FMM order = {order}")
    print("--------------------------------")

    force_i, force_i_exact, angle = run_fmm(x_i, x_j, s_A, s_B, m_j, order, do_print=True)
    ax, rel_error, abs_error = plot_fmm_case(
        s_A, box_A_size, x_i, s_B, box_B_size, x_j, force_i, force_i_exact, 0.2
    )

    plt.title(f"FMM order={order} angle={angle:.5f} rel error={rel_error:.2e}")
    plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
    plt.show()

    print("force_i =", force_i)
    print("force_i_exact =", force_i_exact)
    print("abs error =", abs_error)
    print("rel error =", rel_error)


# %%
# Sweeping through angles
# ^^^^^^^^^^^^^^^^^^^^^^^

s_B = (0, 0, 0)
s_A_all = [(0.8, 0, 0), (1, 0, 0), (1.2, 0, 0)]

box_B_size = 0.5
box_A_size = 0.5

x_j = (0.2, 0.2, 0.0)
x_i = (0.8, 0.2, 0.2)
m_j = 1
m_i = 1

order = 3

for s_A in s_A_all:
    print("--------------------------------")
    print(f"Running FMM s_a = {s_A}")
    print("--------------------------------")

    force_i, force_i_exact, angle = run_fmm(x_i, x_j, s_A, s_B, m_j, order, do_print=True)
    ax, rel_error, abs_error = plot_fmm_case(
        s_A, box_A_size, x_i, s_B, box_B_size, x_j, force_i, force_i_exact, 0.2
    )

    plt.title(f"FMM order={order} angle={angle:.5f} rel error={rel_error:.2e}")
    plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
    plt.show()

    print("force_i =", force_i)
    print("force_i_exact =", force_i_exact)
    print("abs error =", abs_error)
    print("rel error =", rel_error)


# %%
# FMM precision (Angle)
# ^^^^^^^^^^^^^^^^^^^^^

# %%
# For this test we will generate a pair of random positions :math:`x_i` and :math:`x_j`.
# Then we will generate two boxes around the positions :math:`s_A` and :math:`s_B` where each is at a distance box_scale_fact from their respective particle.
# We then perform the FMM expansion to compute the force on :math:`x_i` as well as the exact force.
# We will then plot the relative error as a function of the angle :math:`\theta = (b_A + b_B) / |\mathbf{s}_A - \mathbf{s}_B|` where :math:`b_A` and :math:`b_B` are the distances from the particle to the box centers.

# sphinx_gallery_thumbnail_number = 14

plt.figure()
for order in range(1, 6):
    print("--------------------------------")
    print(f"Running FMM order = {order}")
    print("--------------------------------")

    # set seed
    rng = np.random.default_rng(111)

    N = 50000

    # generate a random set of position in a box of bounds (-1,1)x(-1,1)x(-1,1)
    x_i_all = []
    for i in range(N):
        x_i_all.append((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))

    # same for x_j
    x_j_all = []
    for i in range(N):
        x_j_all.append((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))

    box_scale_fact_all = np.linspace(0, 0.1, N).tolist()

    # same for box_1_center
    s_A_all = []
    for p, box_scale_fact in zip(x_i_all, box_scale_fact_all):
        s_A_all.append(
            (
                p[0] + box_scale_fact * rng.uniform(-1, 1),
                p[1] + box_scale_fact * rng.uniform(-1, 1),
                p[2] + box_scale_fact * rng.uniform(-1, 1),
            )
        )

    # same for box_2_center
    s_B_all = []
    for p, box_scale_fact in zip(x_j_all, box_scale_fact_all):
        s_B_all.append(
            (
                p[0] + box_scale_fact * rng.uniform(-1, 1),
                p[1] + box_scale_fact * rng.uniform(-1, 1),
                p[2] + box_scale_fact * rng.uniform(-1, 1),
            )
        )

    angles = []
    rel_errors = []

    for x_i, x_j, s_A, s_B in zip(x_i_all, x_j_all, s_A_all, s_B_all):

        force_i, force_i_exact, angle = run_fmm(x_i, x_j, s_A, s_B, m_j, order, do_print=False)

        abs_error = np.linalg.norm(force_i - force_i_exact)
        rel_error = abs_error / np.linalg.norm(force_i_exact)

        b_A_size = np.linalg.norm(np.array(s_A) - np.array(x_i))
        b_B_size = np.linalg.norm(np.array(s_B) - np.array(x_j))
        b_dist = np.linalg.norm(np.array(s_A) - np.array(s_B))
        angle = (b_A_size + b_B_size) / b_dist

        if angle > 5.0 or angle < 1e-4:
            continue

        angles.append(angle)
        rel_errors.append(rel_error)

    print(f"Computed for {len(angles)} cases")

    plt.scatter(angles, rel_errors, s=1, label=f"FMM order = {order}")


def plot_powerlaw(order, center_y):
    X = [1e-3, 1e-2 / 3, 1e-1]
    Y = [center_y * (x) ** order for x in X]
    plt.plot(X, Y, linestyle="dashed", color="black")
    bbox = dict(boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.9)
    plt.text(X[1], Y[1], f"$\\propto x^{order}$", fontsize=9, bbox=bbox)


plot_powerlaw(1, 1)
plot_powerlaw(2, 1)
plot_powerlaw(3, 1)
plot_powerlaw(4, 1)
plot_powerlaw(5, 1)

plt.xlabel("Angle")
plt.ylabel("Relative Error")
plt.xscale("log")
plt.yscale("log")
plt.title("FMM precision")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# %%
# Mass moment offset
# ^^^^^^^^^^^^^^^^^^
#
# Now that we know how to compute a FMM force, we now need some remaining
# tools to exploit it fully in a code. In a code using a tree the procedure
# to using a FMM is to first propagate the mass moment upward from leafs
# cells up to the root. Then compute the gravitation moments for all
# cell-cell interations and then propagate the gravitational moment downward
# down to the leaves.
#
# We start with the upward pass for the mass moment. To perform it we need
# to compute the mass moment of a parent according to the one of its children.
# The issue is that the childrens and the parents do not share the same center.
# Therefor we need to offset the mass moment of the children to the parent
# center before summing their moments to get the parent's one.
#
# This is what we call mass moment translation/offset. This section will
# showcase its usage and precision.
#
# We start of by defining a particle :math:`x_j` and a box :math:`s_B` around
# it as well as a new box :math:`s_B'`. The goal will be to offset the mass
# moment of the box :math:`s_B` to the box :math:`s_B'` and compare it to
# the moment of the box :math:`s_B'` computed directly. This should yield
# the same result meaning that we never need to compute the moment directly
# at the parent center and simply use its childrens instead.

s_B = (0, 0, 0)
box_B_size = 1.0
x_j = (0.2, 0.2, 0.0)
m_j = 1

s_B_new = (0.3, 0.3, 0.3)


# %%
# .. raw:: html
#
#   <details>
#   <summary><a>def plot_mass_moment_offset(s_B, s_B_new, box_B_size):</a></summary>
#
def plot_mass_moment_offset(s_B, s_B_new, box_B_size):
    box_B = shamrock.math.AABB_f64_3(
        (
            s_B[0] - box_B_size / 2,
            s_B[1] - box_B_size / 2,
            s_B[2] - box_B_size / 2,
        ),
        (
            s_B[0] + box_B_size / 2,
            s_B[1] + box_B_size / 2,
            s_B[2] + box_B_size / 2,
        ),
    )

    box_B_new = shamrock.math.AABB_f64_3(
        (
            s_B_new[0] - box_B_size / 2,
            s_B_new[1] - box_B_size / 2,
            s_B_new[2] - box_B_size / 2,
        ),
        (
            s_B_new[0] + box_B_size / 2,
            s_B_new[1] + box_B_size / 2,
            s_B_new[2] + box_B_size / 2,
        ),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_arrow(ax, s_B, x_j, "black", "$b_j = x_j - s_B$")
    draw_arrow(ax, s_B_new, x_j, "red", "$b_j' = x_j - s_B'$")

    ax.scatter(s_B[0], s_B[1], s_B[2], color="black", label="s_B")
    ax.scatter(s_B_new[0], s_B_new[1], s_B_new[2], color="red", label="s_B'")
    ax.scatter(x_j[0], x_j[1], x_j[2], color="blue", label="$x_j$")

    draw_aabb(ax, box_B, "blue", 0.2)
    draw_aabb(ax, box_B_new, "red", 0.2)

    center_view = (0.0, 0.0, 0.0)
    view_size = 2.0
    ax.set_xlim(center_view[0] - view_size / 2, center_view[0] + view_size / 2)
    ax.set_ylim(center_view[1] - view_size / 2, center_view[1] + view_size / 2)
    ax.set_zlim(center_view[2] - view_size / 2, center_view[2] + view_size / 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax


# %%
# .. raw:: html
#
#   </details>

# %%

plot_mass_moment_offset(s_B, s_B_new, box_B_size)

plt.title("Mass moment offset illustration")
plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
plt.show()

# %%
# Moment for box B
b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])
Q_n_B = shamrock.math.SymTensorCollection_f64_0_5.from_vec(b_j)
Q_n_B *= m_j
print("b_j =", b_j)
print("Q_n_B =", Q_n_B)

# %%
# Moment for box B'
b_jp = (x_j[0] - s_B_new[0], x_j[1] - s_B_new[1], x_j[2] - s_B_new[2])
Q_n_Bp = shamrock.math.SymTensorCollection_f64_0_5.from_vec(b_jp)
Q_n_Bp *= m_j
print("b_jp =", b_jp)
print("Q_n_Bp =", Q_n_Bp)

# %%
# Offset the moment in box B to box B'
Q_n_B_offset = shamrock.phys.offset_multipole_5(Q_n_B, s_B, s_B_new)
print("Q_n_B_offset =", Q_n_B_offset)

# %%
# Print the norm of the moment in box B'


def tensor_collect_norm(d):
    # detect the type of the tensor collection
    if isinstance(d, shamrock.math.SymTensorCollection_f64_0_5):
        return (
            np.sqrt(d.t0 * d.t0)
            + np.sqrt(d.t1.inner(d.t1))
            + np.sqrt(d.t2.inner(d.t2)) / 2
            + np.sqrt(d.t3.inner(d.t3)) / 6
            + np.sqrt(d.t4.inner(d.t4)) / 24
            + np.sqrt(d.t5.inner(d.t5)) / 120
        )
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_0_4):
        return (
            np.sqrt(d.t0 * d.t0)
            + np.sqrt(d.t1.inner(d.t1))
            + np.sqrt(d.t2.inner(d.t2)) / 2
            + np.sqrt(d.t3.inner(d.t3)) / 6
            + np.sqrt(d.t4.inner(d.t4)) / 24
        )
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_0_3):
        return (
            np.sqrt(d.t0 * d.t0)
            + np.sqrt(d.t1.inner(d.t1))
            + np.sqrt(d.t2.inner(d.t2)) / 2
            + np.sqrt(d.t3.inner(d.t3)) / 6
        )
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_0_2):
        return np.sqrt(d.t0 * d.t0) + np.sqrt(d.t1.inner(d.t1)) + np.sqrt(d.t2.inner(d.t2)) / 2
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_0_1):
        return np.sqrt(d.t0 * d.t0) + np.sqrt(d.t1.inner(d.t1))
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_0_0):
        return np.sqrt(d.t0 * d.t0)
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_1_5):
        return (
            np.sqrt(d.t1.inner(d.t1))
            + np.sqrt(d.t2.inner(d.t2)) / 2
            + np.sqrt(d.t3.inner(d.t3)) / 6
            + np.sqrt(d.t4.inner(d.t4)) / 24
            + np.sqrt(d.t5.inner(d.t5)) / 120
        )
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_1_4):
        return (
            np.sqrt(d.t1.inner(d.t1))
            + np.sqrt(d.t2.inner(d.t2)) / 2
            + np.sqrt(d.t3.inner(d.t3)) / 6
            + np.sqrt(d.t4.inner(d.t4)) / 24
        )
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_1_3):
        return (
            np.sqrt(d.t1.inner(d.t1))
            + np.sqrt(d.t2.inner(d.t2)) / 2
            + np.sqrt(d.t3.inner(d.t3)) / 6
        )
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_1_2):
        return np.sqrt(d.t1.inner(d.t1)) + np.sqrt(d.t2.inner(d.t2)) / 2
    elif isinstance(d, shamrock.math.SymTensorCollection_f64_1_1):
        return np.sqrt(d.t1.inner(d.t1))
    else:
        raise ValueError(f"Unsupported tensor collection type: {type(d)}")


print("Q_n_B norm =", tensor_collect_norm(Q_n_B))
print("Q_n_Bp norm =", tensor_collect_norm(Q_n_Bp))

# %%
# Compute the delta between the moments
delta = Q_n_B_offset - Q_n_Bp
print("delta =", delta)


sqdist_t0 = delta.t0 * delta.t0
sqdist_t1 = delta.t1.inner(delta.t1)
sqdist_t2 = delta.t2.inner(delta.t2)
sqdist_t3 = delta.t3.inner(delta.t3)
sqdist_t4 = delta.t4.inner(delta.t4)
sqdist_t5 = delta.t5.inner(delta.t5)
print("sqdist_t0 =", sqdist_t0)
print("sqdist_t1 =", sqdist_t1)
print("sqdist_t2 =", sqdist_t2)
print("sqdist_t3 =", sqdist_t3)
print("sqdist_t4 =", sqdist_t4)
print("sqdist_t5 =", sqdist_t5)

norm_delta = (
    np.sqrt(sqdist_t0)
    + np.sqrt(sqdist_t1)
    + np.sqrt(sqdist_t2) / 2
    + np.sqrt(sqdist_t3) / 6
    + np.sqrt(sqdist_t4) / 24
    + np.sqrt(sqdist_t5) / 120
)
print("norm_delta =", norm_delta)

print("rel error =", tensor_collect_norm(delta) / tensor_collect_norm(Q_n_Bp))

# %%
# We now want to explore the precision of the offset as a function of the order & distance

plt.figure()

for order in range(0, 6):
    # set seed
    rng = np.random.default_rng(111)

    N = 50000

    # generate a random set of position in a box of bounds (-1,1)x(-1,1)x(-1,1)
    x_j_all = []
    for i in range(N):
        x_j_all.append((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))

    box_scale_fact_all = np.linspace(0, 1, N).tolist()

    # same for box_1_center
    s_B_all = []
    for p, box_scale_fact in zip(x_j_all, box_scale_fact_all):
        s_B_all.append(
            (
                p[0] + box_scale_fact * rng.uniform(-1, 1),
                p[1] + box_scale_fact * rng.uniform(-1, 1),
                p[2] + box_scale_fact * rng.uniform(-1, 1),
            )
        )

    # same for box_2_center
    s_Bp_all = []
    for p, box_scale_fact in zip(x_j_all, box_scale_fact_all):
        s_Bp_all.append(
            (
                p[0] + box_scale_fact * rng.uniform(-1, 1),
                p[1] + box_scale_fact * rng.uniform(-1, 1),
                p[2] + box_scale_fact * rng.uniform(-1, 1),
            )
        )

    center_distances = []
    rel_errors = []
    for x_j, s_B, s_Bp in zip(x_j_all, s_B_all, s_Bp_all):

        b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])
        b_jp = (x_j[0] - s_Bp[0], x_j[1] - s_Bp[1], x_j[2] - s_Bp[2])

        if order == 5:
            Q_n_B = shamrock.math.SymTensorCollection_f64_0_5.from_vec(b_j)
            Q_n_B *= m_j

            Q_n_Bp = shamrock.math.SymTensorCollection_f64_0_5.from_vec(b_jp)
            Q_n_Bp *= m_j

            Q_n_B_offset = shamrock.phys.offset_multipole_5(Q_n_B, s_B, s_Bp)
        elif order == 4:
            Q_n_B = shamrock.math.SymTensorCollection_f64_0_4.from_vec(b_j)
            Q_n_B *= m_j

            Q_n_Bp = shamrock.math.SymTensorCollection_f64_0_4.from_vec(b_jp)
            Q_n_Bp *= m_j

            Q_n_B_offset = shamrock.phys.offset_multipole_4(Q_n_B, s_B, s_Bp)
        elif order == 3:
            Q_n_B = shamrock.math.SymTensorCollection_f64_0_3.from_vec(b_j)
            Q_n_B *= m_j

            Q_n_Bp = shamrock.math.SymTensorCollection_f64_0_3.from_vec(b_jp)
            Q_n_Bp *= m_j

            Q_n_B_offset = shamrock.phys.offset_multipole_3(Q_n_B, s_B, s_Bp)
        elif order == 2:
            Q_n_B = shamrock.math.SymTensorCollection_f64_0_2.from_vec(b_j)
            Q_n_B *= m_j

            Q_n_Bp = shamrock.math.SymTensorCollection_f64_0_2.from_vec(b_jp)
            Q_n_Bp *= m_j

            Q_n_B_offset = shamrock.phys.offset_multipole_2(Q_n_B, s_B, s_Bp)
        elif order == 1:
            Q_n_B = shamrock.math.SymTensorCollection_f64_0_1.from_vec(b_j)
            Q_n_B *= m_j

            Q_n_Bp = shamrock.math.SymTensorCollection_f64_0_1.from_vec(b_jp)
            Q_n_Bp *= m_j

            Q_n_B_offset = shamrock.phys.offset_multipole_1(Q_n_B, s_B, s_Bp)
        elif order == 0:
            Q_n_B = shamrock.math.SymTensorCollection_f64_0_0.from_vec(b_j)
            Q_n_B *= m_j

            Q_n_Bp = shamrock.math.SymTensorCollection_f64_0_0.from_vec(b_jp)
            Q_n_Bp *= m_j

            Q_n_B_offset = shamrock.phys.offset_multipole_0(Q_n_B, s_B, s_Bp)
        else:
            raise ValueError(f"Unsupported offset order: {order}")

        delta = Q_n_B_offset - Q_n_Bp

        rel_error = tensor_collect_norm(delta) / tensor_collect_norm(Q_n_B)
        rel_errors.append(rel_error)

        center_distances.append(np.linalg.norm(np.array(s_B) - np.array(s_Bp)))

    plt.scatter(center_distances, rel_errors, s=1, label=f"multipole order = {order}")

plt.xlabel("$\\vert \\vert s_B - s_B'\\vert \\vert$")
plt.ylabel(
    "$\\vert \\vert Q_n(s_B) - Q_n(s_B') \\vert \\vert / \\vert \\vert Q_n(s_B) \\vert \\vert$ (relative error) "
)
plt.xscale("log")
plt.yscale("log")
plt.title("Mass moment offset precision")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# %%
# As shown the precision is basically the floating point precision.
# Also as a result we can observe a small precision loss for high orders.

# %%
# Gravitational moment offset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Now that we know how to offset the mass moment, we need to offset the gravitational moment.
# This is required as we will compute gravitational moments for cell-cell interactions,
# but we still need to propagate that moment from a parent cell to its children until
# each leaves contains the complete gravitational moment which will be used to compute
# the resulting force.

# %%
# We devise a similar setup to the mass moment offset. We define a particle
# :math:`x_j` and a box of center :math:`s_B` around it. We then define a box
# of center :math:`s_A` around the particle :math:`x_i` as well as a new box
# of center :math:`s_A'`.
#
# The goal will be to offset the gravitational moment of the box :math:`s_A`
# to the box :math:`s_A'` and then compute the resulting FMM force on
# :math:`x_i` in the new box and compare it to the force given the FMM in the
# box :math:`s_A`. If everything is working correctly they should be equals.


def plot_grav_moment_offset(s_A, s_Ap, s_B, box_A_size, box_B_size, x_j):
    box_A = shamrock.math.AABB_f64_3(
        (
            s_A[0] - box_A_size / 2,
            s_A[1] - box_A_size / 2,
            s_A[2] - box_A_size / 2,
        ),
        (
            s_A[0] + box_A_size / 2,
            s_A[1] + box_A_size / 2,
            s_A[2] + box_A_size / 2,
        ),
    )

    box_Ap = shamrock.math.AABB_f64_3(
        (
            s_Ap[0] - box_A_size / 2,
            s_Ap[1] - box_A_size / 2,
            s_Ap[2] - box_A_size / 2,
        ),
        (
            s_Ap[0] + box_A_size / 2,
            s_Ap[1] + box_A_size / 2,
            s_Ap[2] + box_A_size / 2,
        ),
    )

    box_B = shamrock.math.AABB_f64_3(
        (
            s_B[0] - box_B_size / 2,
            s_B[1] - box_B_size / 2,
            s_B[2] - box_B_size / 2,
        ),
        (
            s_B[0] + box_B_size / 2,
            s_B[1] + box_B_size / 2,
            s_B[2] + box_B_size / 2,
        ),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_arrow(ax, s_A, s_B, "purple", "$r = s_B - s_A$")
    draw_arrow(ax, s_Ap, s_B, "purple", "$r' = s_B - s_A'$")

    ax.scatter(s_A[0], s_A[1], s_A[2], color="black", label="s_A")
    ax.scatter(s_Ap[0], s_Ap[1], s_Ap[2], color="black", label="s_Ap")
    ax.scatter(s_B[0], s_B[1], s_B[2], color="black", label="s_B")

    draw_arrow(ax, s_B, x_j, "black", "$b_j = x_j - s_B$")

    ax.scatter(x_j[0], x_j[1], x_j[2], color="red", label="$x_j$")

    draw_aabb(ax, box_A, "blue", 0.1)
    draw_aabb(ax, box_Ap, "cyan", 0.1)
    draw_aabb(ax, box_B, "red", 0.1)

    center_view = (0.5, 0.0, 0.0)
    view_size = 2.0
    ax.set_xlim(center_view[0] - view_size / 2, center_view[0] + view_size / 2)
    ax.set_ylim(center_view[1] - view_size / 2, center_view[1] + view_size / 2)
    ax.set_zlim(center_view[2] - view_size / 2, center_view[2] + view_size / 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax, rel_error, abs_error


s_B = (0, -0.2, -0.2)
s_A = (1, 0, 0)
s_Ap = (1.1, 0.1, 0.0)

box_B_size = 0.5
box_A_size = 0.5

x_j = (0.2, 0.0, -0.5)
x_i = (1.2, 0.2, 0.0)
m_j = 1
m_i = 1

plot_grav_moment_offset(s_A, s_Ap, s_B, box_A_size, box_B_size, x_j)

plt.title("Mass moment offset illustration")
plt.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
plt.show()

b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])
r = (s_B[0] - s_A[0], s_B[1] - s_A[1], s_B[2] - s_A[2])
rp = (s_B[0] - s_Ap[0], s_B[1] - s_Ap[1], s_B[2] - s_Ap[2])


# %%
Q_n_B = shamrock.math.SymTensorCollection_f64_0_4.from_vec(b_j)
Q_n_B *= m_j
print("Q_n_B =", Q_n_B)

# %%
D_n = shamrock.phys.green_func_grav_cartesian_1_5(r)
dM_k = shamrock.phys.get_dM_mat_5(D_n, Q_n_B)
# print("D_n =",D_n)
print("dM_k =", dM_k)

# %%
Dp_n = shamrock.phys.green_func_grav_cartesian_1_5(rp)
dMp_k = shamrock.phys.get_dM_mat_5(Dp_n, Q_n_B)
# print("Dp_n =",Dp_n)
print("dMp_k =", dMp_k)

# %%
# Offset the grav moment to dMp_k
dM_k_offset = shamrock.phys.offset_dM_mat_5(dM_k, s_A, s_Ap)
print("dM_k_offset =", dM_k_offset)

# %%
# Weirdly we can see that for dMk are different even though they will be contracted with the same a_k
# This is normal because we translate the moment dMk into the box A', so even if we estimate the
# force in A' after the translation we will still get the same force as the one we had in A before the translation.
# Which is arguably what we want XD.
delta = dM_k_offset - dMp_k

print("delta =", delta)
print("sqdist_t1 =", np.sqrt(delta.t1.inner(delta.t1)))
print("sqdist_t2 =", np.sqrt(delta.t2.inner(delta.t2)) / 2)
print("sqdist_t3 =", np.sqrt(delta.t3.inner(delta.t3)) / 6)
print("sqdist_t4 =", np.sqrt(delta.t4.inner(delta.t4)) / 24)
print("sqdist_t5 =", np.sqrt(delta.t5.inner(delta.t5)) / 120)
print("(norm) =", tensor_collect_norm(delta))


# %%
a_i = (x_i[0] - s_A[0], x_i[1] - s_A[1], x_i[2] - s_A[2])
a_ip = (x_i[0] - s_Ap[0], x_i[1] - s_Ap[1], x_i[2] - s_Ap[2])

a_k = shamrock.math.SymTensorCollection_f64_0_4.from_vec(a_i)
a_kp = shamrock.math.SymTensorCollection_f64_0_4.from_vec(a_ip)

print("a_k  =", a_k)
print("a_kp =", a_kp)

# %%
result = shamrock.phys.contract_grav_moment_to_force_5(a_k, dM_k)
resultp = shamrock.phys.contract_grav_moment_to_force_5(a_kp, dMp_k)
result_offset = shamrock.phys.contract_grav_moment_to_force_5(a_kp, dM_k_offset)

print("force_i         =", -Gconst * np.array(result))
print("force_ip        =", -Gconst * np.array(resultp))
print("force_ip_offset =", -Gconst * np.array(result_offset), "force_i translated to A'")

# %%
# As expected the delta is almost null
delta_f = np.linalg.norm(np.array(result_offset) - np.array(result))
delta_f /= np.linalg.norm(np.array(result))
print("delta_f =", delta_f)


# %%
# Let's modify FMM in a box to add the translation of the box A
def test_grav_moment_offset(x_i, x_j, s_A, s_Ap, s_B, m_j, order, do_print):

    b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])
    r = (s_B[0] - s_A[0], s_B[1] - s_A[1], s_B[2] - s_A[2])
    a_i = (x_i[0] - s_A[0], x_i[1] - s_A[1], x_i[2] - s_A[2])
    a_ip = (x_i[0] - s_Ap[0], x_i[1] - s_Ap[1], x_i[2] - s_Ap[2])

    if do_print:
        print("x_i =", x_i)
        print("x_j =", x_j)
        print("s_A =", s_A)
        print("s_Ap =", s_Ap)
        print("s_B =", s_B)
        print("b_j =", b_j)
        print("r =", r)
        print("a_i =", a_i)

    # compute the tensor product of the displacment
    if order == 1:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_0.from_vec(b_j)
    elif order == 2:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_1.from_vec(b_j)
    elif order == 3:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_2.from_vec(b_j)
    elif order == 4:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_3.from_vec(b_j)
    elif order == 5:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_4.from_vec(b_j)
    else:
        raise ValueError("Invalid order")

    # multiply by mass to get the mass moment
    Q_n_B *= m_j

    if do_print:
        print("Q_n_B =", Q_n_B)

    # green function gradients
    if order == 1:
        D_n = shamrock.phys.green_func_grav_cartesian_1_1(r)
    elif order == 2:
        D_n = shamrock.phys.green_func_grav_cartesian_1_2(r)
    elif order == 3:
        D_n = shamrock.phys.green_func_grav_cartesian_1_3(r)
    elif order == 4:
        D_n = shamrock.phys.green_func_grav_cartesian_1_4(r)
    elif order == 5:
        D_n = shamrock.phys.green_func_grav_cartesian_1_5(r)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("D_n =", D_n)

    if order == 1:
        dM_k = shamrock.phys.get_dM_mat_1(D_n, Q_n_B)
    elif order == 2:
        dM_k = shamrock.phys.get_dM_mat_2(D_n, Q_n_B)
    elif order == 3:
        dM_k = shamrock.phys.get_dM_mat_3(D_n, Q_n_B)
    elif order == 4:
        dM_k = shamrock.phys.get_dM_mat_4(D_n, Q_n_B)
    elif order == 5:
        dM_k = shamrock.phys.get_dM_mat_5(D_n, Q_n_B)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("dM_k =", dM_k)

    if order == 5:
        dM_k_offset = shamrock.phys.offset_dM_mat_5(dM_k, s_A, s_Ap)
    elif order == 4:
        dM_k_offset = shamrock.phys.offset_dM_mat_4(dM_k, s_A, s_Ap)
    elif order == 3:
        dM_k_offset = shamrock.phys.offset_dM_mat_3(dM_k, s_A, s_Ap)
    elif order == 2:
        dM_k_offset = shamrock.phys.offset_dM_mat_2(dM_k, s_A, s_Ap)
    elif order == 1:
        dM_k_offset = shamrock.phys.offset_dM_mat_1(dM_k, s_A, s_Ap)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("dM_k_offset =", dM_k_offset)

    if order == 1:
        a_k = shamrock.math.SymTensorCollection_f64_0_0.from_vec(a_i)
    elif order == 2:
        a_k = shamrock.math.SymTensorCollection_f64_0_1.from_vec(a_i)
    elif order == 3:
        a_k = shamrock.math.SymTensorCollection_f64_0_2.from_vec(a_i)
    elif order == 4:
        a_k = shamrock.math.SymTensorCollection_f64_0_3.from_vec(a_i)
    elif order == 5:
        a_k = shamrock.math.SymTensorCollection_f64_0_4.from_vec(a_i)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("a_k =", a_k)

    if order == 1:
        a_kp = shamrock.math.SymTensorCollection_f64_0_0.from_vec(a_ip)
    elif order == 2:
        a_kp = shamrock.math.SymTensorCollection_f64_0_1.from_vec(a_ip)
    elif order == 3:
        a_kp = shamrock.math.SymTensorCollection_f64_0_2.from_vec(a_ip)
    elif order == 4:
        a_kp = shamrock.math.SymTensorCollection_f64_0_3.from_vec(a_ip)
    elif order == 5:
        a_kp = shamrock.math.SymTensorCollection_f64_0_4.from_vec(a_ip)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("a_kp =", a_kp)

    if order == 1:
        result = shamrock.phys.contract_grav_moment_to_force_1(a_k, dM_k)
    elif order == 2:
        result = shamrock.phys.contract_grav_moment_to_force_2(a_k, dM_k)
    elif order == 3:
        result = shamrock.phys.contract_grav_moment_to_force_3(a_k, dM_k)
    elif order == 4:
        result = shamrock.phys.contract_grav_moment_to_force_4(a_k, dM_k)
    elif order == 5:
        result = shamrock.phys.contract_grav_moment_to_force_5(a_k, dM_k)
    else:
        raise ValueError("Invalid order")

    Gconst = 1  # let's just set the grav constant to 1
    force_i = -Gconst * np.array(result)
    if do_print:
        print("force_i =", force_i)

    if order == 1:
        result_offset = shamrock.phys.contract_grav_moment_to_force_1(a_kp, dM_k_offset)
    elif order == 2:
        result_offset = shamrock.phys.contract_grav_moment_to_force_2(a_kp, dM_k_offset)
    elif order == 3:
        result_offset = shamrock.phys.contract_grav_moment_to_force_3(a_kp, dM_k_offset)
    elif order == 4:
        result_offset = shamrock.phys.contract_grav_moment_to_force_4(a_kp, dM_k_offset)
    elif order == 5:
        result_offset = shamrock.phys.contract_grav_moment_to_force_5(a_kp, dM_k_offset)
    else:
        raise ValueError("Invalid order")

    force_i_offset = -Gconst * np.array(result_offset)
    if do_print:
        print("force_i_offset =", force_i_offset)

    b_A_size = np.linalg.norm(np.array(s_A) - np.array(x_i))
    b_B_size = np.linalg.norm(np.array(s_B) - np.array(x_j))
    b_dist = np.linalg.norm(np.array(s_A) - np.array(s_B))

    angle = (b_A_size + b_B_size) / b_dist

    delta_A = np.linalg.norm(np.array(s_A) - np.array(s_Ap))

    if do_print:
        print("b_A_size =", b_A_size)
        print("b_B_size =", b_B_size)
        print("b_dist =", b_dist)
        print("angle =", angle)

    return force_i, force_i_offset, angle, delta_A


# %%
# Let test for many different parameters.
# For clarification a perfect result here is that the translated dMk contracted with the new displacment ak_p give the same result as the original expansion (which it does ;) ).

plt.figure()
for order in range(1, 6):
    print("--------------------------------")
    print(f"Running FMM order = {order}")
    print("--------------------------------")

    # set seed
    rng = np.random.default_rng(111)

    N = 50000

    # generate a random set of position in a box of bounds (-1,1)x(-1,1)x(-1,1)
    x_i_all = []
    for i in range(N):
        x_i_all.append((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))

    # same for x_j
    x_j_all = []
    for i in range(N):
        x_j_all.append((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))

    box_scale_fact_all = np.linspace(0, 0.1, N).tolist()

    # same for box_1_center
    s_A_all = []
    s_Ap_all = []
    for p, box_scale_fact in zip(x_i_all, box_scale_fact_all):
        s_A_all.append(
            (
                p[0] + box_scale_fact * rng.uniform(-1, 1),
                p[1] + box_scale_fact * rng.uniform(-1, 1),
                p[2] + box_scale_fact * rng.uniform(-1, 1),
            )
        )
        s_Ap_all.append(
            (
                p[0] + box_scale_fact * rng.uniform(-1, 1),
                p[1] + box_scale_fact * rng.uniform(-1, 1),
                p[2] + box_scale_fact * rng.uniform(-1, 1),
            )
        )

    # same for box_2_center
    s_B_all = []
    for p, box_scale_fact in zip(x_j_all, box_scale_fact_all):
        s_B_all.append(
            (
                p[0] + box_scale_fact * rng.uniform(-1, 1),
                p[1] + box_scale_fact * rng.uniform(-1, 1),
                p[2] + box_scale_fact * rng.uniform(-1, 1),
            )
        )

    angles = []
    delta_A_all = []
    rel_errors = []

    for x_i, x_j, s_A, s_Ap, s_B in zip(x_i_all, x_j_all, s_A_all, s_Ap_all, s_B_all):

        force_i, force_i_offset, angle, delta_A = test_grav_moment_offset(
            x_i, x_j, s_A, s_Ap, s_B, m_j, order, do_print=False
        )

        abs_error = np.linalg.norm(force_i_offset - force_i)
        rel_error = abs_error / np.linalg.norm(force_i)

        b_A_size = np.linalg.norm(np.array(s_A) - np.array(x_i))
        b_B_size = np.linalg.norm(np.array(s_B) - np.array(x_j))
        b_dist = np.linalg.norm(np.array(s_A) - np.array(s_B))
        angle = (b_A_size + b_B_size) / b_dist

        if angle > 5.0 or angle < 1e-4:
            continue

        angles.append(angle)
        delta_A_all.append(delta_A)
        rel_errors.append(rel_error)

    print(f"Computed for {len(angles)} cases")

    plt.scatter(angles, rel_errors, s=1, label=f"FMM order = {order}")


plt.xlabel("Angle")
plt.ylabel("$|f_{\\rm fmm} - f_{\\rm fmm, offset}| / |f_{\\rm fmm}|$ (Relative error)")
plt.xscale("log")
plt.yscale("log")
plt.title("Grav moment translation error")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# %%
# Small note on multipole method (Without FMM)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# %%
# In some ways a MM method is a FMM where there is no box A.
# If we reuse the formula at the start of this page we get:
#
# .. math::
#    \mathbf{b}_j = \mathbf{x}_j - \mathbf{s}_b
#
# and the distance between the centers of the boxes is:
#
# .. math::
#    \mathbf{r} = \mathbf{s}_b - \mathbf{x}_i
#
# This implies that the distance between the two particles is:
#
# .. math::
#    \mathbf{x}_j - \mathbf{x}_i = \mathbf{r} + \mathbf{b}_j

# %%
# .. math::
#    \phi(\mathbf{x}_i) &= - \mathcal{G}\iiint_V \rho(\mathbf{x}_j) G(\mathbf{x}_j - \mathbf{x}_i) ~{\rm d}^3\mathbf{x}_j\\
#    &\simeq - \mathcal{G}\iiint_V \rho(\mathbf{x}_j)  \sum_{n = 0}^p \frac{1}{n!} \nabla_r^{(n)} G(\mathbf{r}) \cdot \mathbf{b}_j^{(n)} ~{\rm d}^3\mathbf{x}_j\\
#    &= - \mathcal{G}\sum_{n = 0}^p \frac{1}{n!} \underbrace{\nabla_r^{(n)} G(\mathbf{r})}_{D_n} \cdot \underbrace{\left(\iiint_V \rho(\mathbf{x}_j) \mathbf{b}_j^{(n)}~{\rm d}^3\mathbf{x}_j\right)}_{Q_n^B},
#
# where :math:`D_n` are the gradients of the Green function and :math:`Q^B_n` are
# the moments of the mass distribution. Hence, the force can then be written as
# follows:
#
# .. math::
#   f_{\rm g}(\mathbf{x}_i) &= -\nabla_i \phi(\mathbf{x}_i)\\
#                           &= \mathcal{G}\iiint_V \rho(\mathbf{x}_j) \nabla_i G(\mathbf{x}_j - \mathbf{x}_i) ~{\rm d}^3\mathbf{x}_j\\
#                           &= - \mathcal{G}\iiint_V \rho(\mathbf{x}_j) \nabla_j G(\mathbf{x}_j - \mathbf{x}_i) ~{\rm d}^3\mathbf{x}_j\\
#                           &= - \mathcal{G}\sum_{n = 0}^p \frac{1}{n!} \nabla_r {D_n} \cdot {Q_n^B}\\
#                           &= -\mathcal{G} \sum_{n = 0}^p \frac{1}{n!} {D_{n+1}} \cdot {Q_n^B} \\
#                           &= -\mathcal{G} \sum_{n = 0}^p \frac{1}{n!} {Q_n^B} \cdot {D_{n+1}}
#
# As we can see, the expression of the MM force is litteraly the same contraction
# as the end of the FMM. Essentially in MM the green function moments are the
# equivalent of :math:`dM_k` in FMM. So we can use the same final function but put the mass moments
# instead of box a displacements.


s_B = (0, 0, 0)

box_B_size = 0.5

x_j = (0.2, 0.2, 0.0)
x_i = (1.2, 0.2, 0.0)
m_j = 1
m_i = 1

b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])
r = (s_B[0] - x_i[0], s_B[1] - x_i[1], s_B[2] - x_i[2])

# %%
Q_n_B = shamrock.math.SymTensorCollection_f64_0_4.from_vec(b_j)
Q_n_B *= m_j
print("Q_n_B =", Q_n_B)

# %%
D_n = shamrock.phys.green_func_grav_cartesian_1_5(r)
print("D_n =", D_n)

# %%
result = shamrock.phys.contract_grav_moment_to_force_5(Q_n_B, D_n)
Gconst = 1  # let's just set the grav constant to 1
force_i = -Gconst * np.array(result)
print("force_i =", force_i)

# %%
# We can check that this is equivalent to the FMM with s_A = (0,0,0)
dM_k = shamrock.phys.get_dM_mat_5(D_n, Q_n_B)
print("dM_k =", dM_k)
a_k = shamrock.math.SymTensorCollection_f64_0_4.from_vec((0, 0, 0))
print("a_k =", a_k)
force_i_fmm_sA_null = shamrock.phys.contract_grav_moment_to_force_5(a_k, dM_k)
Gconst = 1  # let's just set the grav constant to 1
force_i_fmm_sA_null = -Gconst * np.array(force_i_fmm_sA_null)
print("force_i_fmm_sA_null =", force_i_fmm_sA_null)


# %%
# Now we just need the analytical force to compare
def analytic_force_i(x_i, x_j, Gconst):
    force_i_direct = (x_j[0] - x_i[0], x_j[1] - x_i[1], x_j[2] - x_i[2])
    force_i_direct /= np.linalg.norm(force_i_direct) ** 3
    force_i_direct *= m_i
    return force_i_direct


force_i_exact = analytic_force_i(x_i, x_j, Gconst)
print("force_i_exact =", force_i_exact)

# %%
abs_error = np.linalg.norm(force_i - force_i_exact)
rel_error = abs_error / np.linalg.norm(force_i)


b_B_size = np.linalg.norm(np.array(s_B) - np.array(x_j))
b_dist = np.linalg.norm(np.array(x_i) - np.array(s_B))
angle = (b_B_size) / b_dist

print("abs_error =", abs_error)
print("rel_error =", rel_error)
print("angle =", angle)

assert rel_error < 1e-2

# %%
# Let's code MM in a box


def run_mm(x_i, x_j, s_B, m_j, order, do_print):

    b_j = (x_j[0] - s_B[0], x_j[1] - s_B[1], x_j[2] - s_B[2])
    r = (s_B[0] - x_i[0], s_B[1] - x_i[1], s_B[2] - x_i[2])

    if do_print:
        print("x_i =", x_i)
        print("x_j =", x_j)
        print("s_B =", s_B)
        print("b_j =", b_j)
        print("r =", r)

    # compute the tensor product of the displacment
    if order == 1:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_0.from_vec(b_j)
    elif order == 2:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_1.from_vec(b_j)
    elif order == 3:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_2.from_vec(b_j)
    elif order == 4:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_3.from_vec(b_j)
    elif order == 5:
        Q_n_B = shamrock.math.SymTensorCollection_f64_0_4.from_vec(b_j)
    else:
        raise ValueError("Invalid order")

    # multiply by mass to get the mass moment
    Q_n_B *= m_j

    if do_print:
        print("Q_n_B =", Q_n_B)

    # green function gradients
    if order == 1:
        D_n = shamrock.phys.green_func_grav_cartesian_1_1(r)
    elif order == 2:
        D_n = shamrock.phys.green_func_grav_cartesian_1_2(r)
    elif order == 3:
        D_n = shamrock.phys.green_func_grav_cartesian_1_3(r)
    elif order == 4:
        D_n = shamrock.phys.green_func_grav_cartesian_1_4(r)
    elif order == 5:
        D_n = shamrock.phys.green_func_grav_cartesian_1_5(r)
    else:
        raise ValueError("Invalid order")

    if do_print:
        print("D_n =", D_n)

    if order == 1:
        result = shamrock.phys.contract_grav_moment_to_force_1(Q_n_B, D_n)
    elif order == 2:
        result = shamrock.phys.contract_grav_moment_to_force_2(Q_n_B, D_n)
    elif order == 3:
        result = shamrock.phys.contract_grav_moment_to_force_3(Q_n_B, D_n)
    elif order == 4:
        result = shamrock.phys.contract_grav_moment_to_force_4(Q_n_B, D_n)
    elif order == 5:
        result = shamrock.phys.contract_grav_moment_to_force_5(Q_n_B, D_n)
    else:
        raise ValueError("Invalid order")

    Gconst = 1  # let's just set the grav constant to 1
    force_i = -Gconst * np.array(result)
    if do_print:
        print("force_i =", force_i)

    force_i_exact = analytic_force_i(x_i, x_j, Gconst)
    if do_print:
        print("force_i_exact =", force_i_exact)

    b_B_size = np.linalg.norm(np.array(s_B) - np.array(x_j))
    b_dist = np.linalg.norm(np.array(x_i) - np.array(s_B))

    angle = (b_B_size) / b_dist

    if do_print:
        print("b_A_size =", b_A_size)
        print("b_B_size =", b_B_size)
        print("b_dist =", b_dist)
        print("angle =", angle)

    return force_i, force_i_exact, angle


# %%

plt.figure()
for order in range(1, 6):
    print("--------------------------------")
    print(f"Running MM order = {order}")
    print("--------------------------------")

    # set seed
    rng = np.random.default_rng(111)

    N = 50000

    # generate a random set of position in a box of bounds (-1,1)x(-1,1)x(-1,1)
    x_i_all = []
    for i in range(N):
        x_i_all.append((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))

    # same for x_j
    x_j_all = []
    for i in range(N):
        x_j_all.append((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))

    box_scale_fact_all = np.linspace(0, 0.1, N).tolist()

    # same for box_2_center
    s_B_all = []
    for p, box_scale_fact in zip(x_j_all, box_scale_fact_all):
        s_B_all.append(
            (
                p[0] + box_scale_fact * rng.uniform(-1, 1),
                p[1] + box_scale_fact * rng.uniform(-1, 1),
                p[2] + box_scale_fact * rng.uniform(-1, 1),
            )
        )

    angles = []
    rel_errors = []

    for x_i, x_j, s_B in zip(x_i_all, x_j_all, s_B_all):

        force_i, force_i_exact, angle = run_mm(x_i, x_j, s_B, m_j, order, do_print=False)

        abs_error = np.linalg.norm(force_i - force_i_exact)
        rel_error = abs_error / np.linalg.norm(force_i_exact)

        if angle > 5.0 or angle < 1e-4:
            continue

        angles.append(angle)
        rel_errors.append(rel_error)

    print(f"Computed for {len(angles)} cases")

    plt.scatter(angles, rel_errors, s=1, label=f"MM order = {order}")


def plot_powerlaw(order, center_y):
    X = [1e-3, 1e-2 / 3, 1e-1]
    Y = [center_y * (x) ** order for x in X]
    plt.plot(X, Y, linestyle="dashed", color="black")
    bbox = dict(boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.9)
    plt.text(X[1], Y[1], f"$\\propto x^{order}$", fontsize=9, bbox=bbox)


plot_powerlaw(1, 1)
plot_powerlaw(2, 1)
plot_powerlaw(3, 1)
plot_powerlaw(4, 1)
plot_powerlaw(5, 1)

plt.xlabel("Angle")
plt.ylabel("Relative Error")
plt.xscale("log")
plt.yscale("log")
plt.title("MM precision")
plt.legend(loc="lower right")
plt.grid()
plt.show()
