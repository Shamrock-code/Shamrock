// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file py_particle_injection.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings of the particle injection, shared by the SPH and GSPH solver configs.
 *
 */

#include "shambase/numeric_limits.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/common/injection/ParticleInjectionConfig.hpp"
#include <cmath>
#include <optional>

namespace shammodels::common::injection {

    /**
     * @brief Add the particle injection methods to the python class of a solver config
     *
     * @tparam TConfig a solver config with a particle_injection member and a Kernel type
     */
    template<class Tvec, class TConfig>
    inline void add_particle_injection_defs(py::class_<TConfig> &cls) {
        using Tscal = shambase::VecComponent<Tvec>;

        cls.def(
            "add_injection_fallback_stream",
            [](TConfig &self,
               const Tvec &inject_pos,
               const Tvec &inject_vel,
               Tscal stream_radius,
               Tscal u_inject,
               Tscal mdot_peak,
               Tscal t_mb,
               Tscal t_disrupt,
               Tscal mdot_index,
               Tscal t_end,
               std::optional<u32> n_boundary_walls) {
                using Kernel = typename TConfig::Kernel;

                // by default, enough walls to cover the kernel support of the youngest wall
                u32 n_walls = n_boundary_walls.value_or(
                    static_cast<u32>(std::ceil(Kernel::Rkern * Kernel::hfactd)) + 1);

                FallbackStream<Tvec> stream{
                    inject_pos,
                    inject_vel,
                    stream_radius,
                    u_inject,
                    {mdot_peak, t_mb, t_disrupt, mdot_index, t_end},
                    n_walls};

                self.particle_injection.add_fallback_stream(stream);
            },
            py::kw_only(),
            py::arg("inject_pos"),
            py::arg("inject_vel"),
            py::arg("stream_radius"),
            py::arg("u_inject"),
            py::arg("mdot_peak"),
            py::arg("t_mb"),
            py::arg("t_disrupt")        = 0.,
            py::arg("mdot_index")       = -5. / 3.,
            py::arg("t_end")            = shambase::get_infty<Tscal>(),
            py::arg("n_boundary_walls") = std::nullopt,
            R"==(
    Inject a stream of gas following a tidal disruption fallback rate.

    The fallback rate is

    .. math::
        \dot M(t) = \dot M_{\rm peak} \left(\frac{t - t_{\rm disrupt}}{t_{\rm mb}}\right)^{n}

    for :math:`t_{\rm disrupt} + t_{\rm mb} \le t < t_{\rm end}`, and zero otherwise.

    The gas crosses a disc of radius ``stream_radius`` centred on ``inject_pos`` and orthogonal
    to ``inject_vel``, as walls of particles (square lattice cropped to the disc). A wall is born
    each time the rate has supplied its mass, so that the mean rate follows :math:`\dot M(t)`
    whatever the timestep, and the lattice spacing follows the stream density
    :math:`\dot M / (\pi R^2 v)` to keep the particle distribution isotropic. The next walls to be
    born are kept upstream of the injection point as boundary particles, re-created every step at
    their analytic position, so that the youngest walls see a correct pressure.

    Injected particles have the particle mass of the simulation and are tagged in the ``part_id``
    field, added to the layout when an injection is configured: bit 62 is set for injected
    particles, bit 61 for boundary particles (filter them out of the analysis with
    ``(part_id >> 61) & 1``). The other particles keep the ids of the setup (0 unless particle
    tracking is enabled). Injection needs no MPI communication, is deterministic, and requires
    experimental features. On restart, the walls born before the restart time are assumed to be
    in the dump.

    The region around the injection point should be free of other gas: the boundary walls are
    re-created there every step, and particles buried in them can make the smoothing length
    iteration diverge (``set_smoothing_length_density_based_bisecting_nr`` is more robust to this).

    Parameters
    ----------
    inject_pos : tuple[float, float, float]
        Centre of the injection disc
    inject_vel : tuple[float, float, float]
        Velocity of the injected gas
    stream_radius : float
        Radius of the stream cross-section
    u_inject : float
        Specific internal energy of the injected gas
    mdot_peak : float
        Rate at the start of the fallback, :math:`t = t_{\rm disrupt} + t_{\rm mb}`
    t_mb : float
        Return time of the most bound debris, counted from ``t_disrupt``
    t_disrupt : float
        Time of the disruption (default 0)
    mdot_index : float
        Power law index :math:`n` of the rate (default -5/3, 0 for a constant rate)
    t_end : float
        Time at which the injection stops (default never)
    n_boundary_walls : int, optional
        Number of walls kept upstream as boundary particles (default: enough walls to cover the
        kernel support of the youngest wall)
)==");
    }

} // namespace shammodels::common::injection
