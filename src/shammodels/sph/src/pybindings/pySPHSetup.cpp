// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHSetup.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::modules::SPHSetup
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/sph/modules/SPHSetup.hpp"
#include "sph_pybindings.hpp"
#include <memory>
#include <optional>
#include <random>

namespace shammodels::sph::pybindings {

    template<class Tvec, template<class> class SPHKernel>
    void add_sph_setup_instance(py::module &m, const std::string &name_model) {
        using namespace shammodels::sph;

        using Tscal = shambase::VecComponent<Tvec>;

        using TSPHSetup = shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>;

        std::string setup_name = name_model + "_SPHSetup";
        py::class_<TSPHSetup>(m, setup_name.c_str())
            .def(
                "make_generator_lattice_hcp",
                [](TSPHSetup &self,
                   Tscal dr,
                   Tvec box_min,
                   Tvec box_max,
                   bool discontinuous,
                   Tscal init_h_factor) {
                    return self.make_generator_lattice_hcp(
                        dr, {box_min, box_max}, discontinuous, init_h_factor);
                },
                py::arg("dr"),
                py::arg("box_min"),
                py::arg("box_max"),
                py::arg("discontinuous") = true,
                py::arg("init_h_factor")
                = modules::GeneratorLatticeHCP<Tvec>::default_init_h_factor,
                R"==(
    Generate particles on a HCP lattice of parameter dr (neighbours are 2 dr apart)

    The initial smoothing length is set to init_h_factor * dr. The default 2^(5/6)
    is the equilibrium smoothing length for hfact = 1 (the smallest hfact of all
    the SPH kernels), so the initial guess never exceeds the equilibrium value.
)==")
            .def(
                "make_generator_lattice_fcc",
                [](TSPHSetup &self,
                   Tscal dr,
                   Tvec box_min,
                   Tvec box_max,
                   bool discontinuous,
                   Tscal init_h_factor) {
                    return self.make_generator_lattice_fcc(
                        dr, {box_min, box_max}, discontinuous, init_h_factor);
                },
                py::arg("dr"),
                py::arg("box_min"),
                py::arg("box_max"),
                py::arg("discontinuous") = true,
                py::arg("init_h_factor")
                = modules::GeneratorLatticeFCC<Tvec>::default_init_h_factor,
                R"==(
    Generate particles on a true FCC lattice (ABC stacking of close-packed layers along z)
    of parameter dr (neighbours are 2 dr apart)

    The number of layers along z must be a multiple of 3, the number of rows along y even

    The initial smoothing length is set to init_h_factor * dr. The default 2^(5/6)
    is the equilibrium smoothing length for hfact = 1 (the smallest hfact of all
    the SPH kernels), so the initial guess never exceeds the equilibrium value.
)==")
            .def(
                "make_generator_lattice_cubic",
                [](TSPHSetup &self, Tscal dr, Tvec box_min, Tvec box_max) {
                    return self.make_generator_lattice_cubic(dr, {box_min, box_max});
                })
            .def(
                "make_generator_disc_mc",
                [](TSPHSetup &self,
                   Tscal part_mass,
                   Tscal disc_mass,
                   Tscal r_in,
                   Tscal r_out,
                   std::function<Tscal(Tscal)> sigma_profile,
                   std::function<Tscal(Tscal)> H_profile,
                   std::function<Tscal(Tscal)> rot_profile,
                   std::function<Tscal(Tscal)> cs_profile,
                   std::function<Tvec(Tvec)> velocity_field,
                   std::function<Tscal(Tvec)> cs_field,
                   u64 random_seed,
                   Tscal init_h_factor) {
                    auto build_vel_lambda = [&]() -> std::function<Tvec(Tvec)> {
                        if (!velocity_field && !rot_profile) {
                            throw shambase::make_except_with_loc<std::invalid_argument>(
                                "make_generator_disc_mc: either velocity_field or rot_profile must "
                                "be "
                                "provided, you must provide one of them");
                        }

                        if (velocity_field && rot_profile) {
                            throw shambase::make_except_with_loc<std::invalid_argument>(
                                "make_generator_disc_mc: either velocity_field or rot_profile must "
                                "be "
                                "provided, you cannot provide both");
                        }

                        if (velocity_field) {
                            return std::move(velocity_field);
                        }
                        return [vth_r = std::move(rot_profile)](Tvec pos) {
                            pos[2]  = 0; // to get the cylindrical radius
                            Tscal r = sycl::length(pos);

                            auto etheta = sycl::vec<Tscal, 3>{-pos.y(), pos.x(), 0};
                            etheta /= sycl::length(etheta);

                            return vth_r(r) * etheta;
                        };
                    };

                    auto build_cs_lambda = [&]() -> std::function<Tscal(Tvec)> {
                        bool need_cs = self.solver_config.is_eos_locally_isothermal();

                        if (!need_cs) {
                            if (cs_field) {
                                if (shamcomm::world_rank() == 0) {
                                    logger::warn_ln(
                                        "SPHSetup",
                                        "make_generator_disc_mc: with the current EOS, cs_field is "
                                        "ignored");
                                }
                            }
                            if (cs_profile) {
                                if (shamcomm::world_rank() == 0) {
                                    logger::warn_ln(
                                        "SPHSetup",
                                        "make_generator_disc_mc: with the current EOS, cs_profile "
                                        "is "
                                        "ignored");
                                }
                            }
                            return std::function<Tscal(Tvec)>{};
                        }

                        if (!cs_field && !cs_profile) {
                            throw shambase::make_except_with_loc<std::invalid_argument>(
                                "make_generator_disc_mc: either cs_field or cs_profile must be "
                                "provided, you must provide one of them");
                        }

                        if (cs_field && cs_profile) {
                            throw shambase::make_except_with_loc<std::invalid_argument>(
                                "make_generator_disc_mc: either cs_field or cs_profile must be "
                                "provided, you cannot provide both");
                        }

                        if (cs_field) {
                            return std::move(cs_field);
                        }

                        return [cs_r = std::move(cs_profile)](Tvec pos) {
                            pos[2]  = 0; // to get the cylindrical radius
                            Tscal r = sycl::length(pos);
                            return cs_r(r);
                        };
                    };

                    return self.make_generator_disc_mc(
                        part_mass,
                        disc_mass,
                        r_in,
                        r_out,
                        std::move(sigma_profile),
                        std::move(H_profile),
                        build_vel_lambda(),
                        build_cs_lambda(),
                        std::mt19937_64(random_seed),
                        init_h_factor);
                },
                py::kw_only(),
                py::arg("part_mass"),
                py::arg("disc_mass"),
                py::arg("r_in"),
                py::arg("r_out"),
                py::arg("sigma_profile"),
                py::arg("H_profile"),
                py::arg("rot_profile")    = std::function<Tscal(Tscal)>{},
                py::arg("cs_profile")     = std::function<Tscal(Tscal)>{},
                py::arg("velocity_field") = std::function<Tvec(Tvec)>{},
                py::arg("cs_field")       = std::function<Tscal(Tvec)>{},
                py::arg("random_seed"),
                py::arg("init_h_factor") = 0.8,
                R"pbdoc(
        Create a Monte Carlo disc particle generator.

        Particles are sampled in cylindrical coordinates: the radius is drawn
        with rejection sampling from ``sigma_profile``, the azimuth is uniform,
        and the vertical coordinate follows a Gaussian with scale ``H_profile(r)``.
        The initial density is extrapolated from the surface density profile, and
        smoothing lengths are set from that density.

        Args:
            part_mass: Mass of each SPH particle.
            disc_mass: Total disc mass. The particle count is ``disc_mass / part_mass``.
            r_in: Inner disc radius.
            r_out: Outer disc radius.
            sigma_profile: Surface density profile ``sigma(r)``.
            H_profile: Disc scale height profile ``H(r)``.
            rot_profile: Azimuthal speed profile ``v_theta(r)``. The velocity is
                projected along the cylindrical azimuthal direction at each
                particle position. Mutually exclusive with ``velocity_field``.
            cs_profile: Sound speed profile ``c_s(r)``. Evaluated at the cylindrical
                radius of each particle. Required when the solver uses a locally
                isothermal EOS. Mutually exclusive with ``cs_field``.
            velocity_field: Velocity profile ``v(x, y, z)``. Mutually exclusive
                with ``rot_profile``.
            cs_field: Sound speed profile ``c_s(x, y, z)``. Required when the solver
                uses a locally isothermal EOS. Mutually exclusive with ``cs_profile``.
            random_seed: Seed for the Monte Carlo sampler.
            init_h_factor: Multiplier applied to the smoothing length inferred from
                the generated density. Defaults to ``0.8``.

        Notes:
            Exactly one of ``velocity_field`` or ``rot_profile`` must be provided.

            If the solver uses a locally isothermal EOS, exactly one of ``cs_field``
            or ``cs_profile`` must be provided. Otherwise both sound-speed profiles
            are ignored and a warning is emitted if either is supplied.

        Returns:
            A setup node to pass to :py:meth:`apply_setup`.
    )pbdoc")
            .def(
                "make_generator_from_context",
                [](TSPHSetup &self, ShamrockCtx &context_other) {
                    return self.make_generator_from_context(context_other);
                })
            .def(
                "make_combiner_add",
                [](TSPHSetup &self,
                   shammodels::sph::modules::SetupNodePtr parent1,
                   shammodels::sph::modules::SetupNodePtr parent2) {
                    return self.make_combiner_add(parent1, parent2);
                })
            .def(
                "make_modifier_warp_disc",
                [](TSPHSetup &self,
                   shammodels::sph::modules::SetupNodePtr parent,
                   Tscal Rwarp,
                   Tscal Hwarp,
                   Tscal inclination,
                   Tscal posangle) {
                    return self.make_modifier_warp_disc(
                        parent, Rwarp, Hwarp, inclination, posangle);
                },
                py::kw_only(),
                py::arg("parent"),
                py::arg("Rwarp"),
                py::arg("Hwarp"),
                py::arg("inclination"),
                py::arg("posangle") = 0.)
            .def(
                "make_modifier_custom_warp",
                [](TSPHSetup &self,
                   shammodels::sph::modules::SetupNodePtr parent,
                   std::function<Tscal(Tscal)> inc_profile,
                   std::function<Tscal(Tscal)> psi_profile,
                   std::function<Tvec(Tscal)> k_profile) {
                    return self.make_modifier_custom_warp(
                        parent, inc_profile, psi_profile, k_profile);
                },
                py::kw_only(),
                py::arg("parent"),
                py::arg("inc_profile"),
                py::arg("psi_profile"),
                py::arg("k_profile"))
            .def(
                "make_modifier_offset",
                [](TSPHSetup &self,
                   shammodels::sph::modules::SetupNodePtr parent,
                   Tvec offset_postion,
                   Tvec offset_velocity) {
                    return self.make_modifier_add_offset(parent, offset_postion, offset_velocity);
                },
                py::kw_only(),
                py::arg("parent"),
                py::arg("offset_position"),
                py::arg("offset_velocity"))
            .def(
                "make_modifier_filter",
                [](TSPHSetup &self,
                   shammodels::sph::modules::SetupNodePtr parent,
                   std::function<bool(Tvec)> filter) {
                    return self.make_modifier_filter(parent, filter);
                },
                py::kw_only(),
                py::arg("parent"),
                py::arg("filter"))
            .def(
                "make_modifier_split_part",
                [](TSPHSetup &self,
                   shammodels::sph::modules::SetupNodePtr parent,
                   u64 n_split,
                   u64 seed,
                   Tscal h_scaling) {
                    return self.make_modifier_split_part(parent, n_split, seed, h_scaling);
                },
                py::kw_only(),
                py::arg("parent"),
                py::arg("n_split"),
                py::arg("seed"),
                py::arg("h_scaling") = 0.6)
            .def(
                "apply_setup",
                [](TSPHSetup &self,
                   shammodels::sph::modules::SetupNodePtr setup,
                   bool part_reordering,
                   std::optional<u32> gen_step,
                   std::optional<u32> insert_step,
                   std::optional<u64> msg_count_limit,
                   std::optional<u64> msg_size_limit,
                   std::optional<u64> max_msg_size,
                   bool do_setup_log,
                   bool use_new_setup,
                   bool speculative_balancing) {
                    if (use_new_setup) {
                        return self.apply_setup_new(
                            setup,
                            part_reordering,
                            gen_step,
                            insert_step,
                            msg_count_limit,
                            msg_size_limit,
                            max_msg_size,
                            do_setup_log,
                            speculative_balancing);
                    } else {
                        if (bool(gen_step)) {
                            ON_RANK_0(
                                logger::warn_ln(
                                    "SPHSetup", "gen_step is ignored when using old setup"));
                        }
                        if (bool(msg_count_limit)) {
                            ON_RANK_0(
                                logger::warn_ln(
                                    "SPHSetup", "msg_count_limit is ignored when using old setup"));
                        }
                        if (bool(msg_size_limit)) {
                            ON_RANK_0(
                                logger::warn_ln(
                                    "SPHSetup", "msg_size_limit is ignored when using old setup"));
                        }
                        if (bool(max_msg_size)) {
                            ON_RANK_0(
                                logger::warn_ln(
                                    "SPHSetup", "max_msg_size is ignored when using old setup"));
                        }
                        if (bool(do_setup_log)) {
                            ON_RANK_0(
                                logger::warn_ln(
                                    "SPHSetup", "do_setup_log is ignored when using old setup"));
                        }
                        return self.apply_setup(setup, part_reordering, insert_step);
                    }
                },
                py::arg("setup"),
                py::kw_only(),
                py::arg("part_reordering")       = true,
                py::arg("gen_step")              = std::nullopt,
                py::arg("insert_step")           = std::nullopt,
                py::arg("msg_count_limit")       = std::nullopt,
                py::arg("rank_comm_size_limit")  = std::nullopt,
                py::arg("max_msg_size")          = std::nullopt,
                py::arg("do_setup_log")          = false,
                py::arg("use_new_setup")         = true,
                py::arg("speculative_balancing") = false);
    }

    template void add_sph_setup_instance<f64_3, shammath::M4>(py::module &, const std::string &);
    template void add_sph_setup_instance<f64_3, shammath::M6>(py::module &, const std::string &);
    template void add_sph_setup_instance<f64_3, shammath::M8>(py::module &, const std::string &);
    template void add_sph_setup_instance<f64_3, shammath::C2>(py::module &, const std::string &);
    template void add_sph_setup_instance<f64_3, shammath::C4>(py::module &, const std::string &);
    template void add_sph_setup_instance<f64_3, shammath::C6>(py::module &, const std::string &);

} // namespace shammodels::sph::pybindings
