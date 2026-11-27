// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/logs/loglevels.hpp"
#include "shambase/memory.hpp"
#include "pybindings/sph_pybindings.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/modules/AnalysisAngularMomentum.hpp"
#include "shammodels/sph/modules/AnalysisBarycenter.hpp"
#include "shammodels/sph/modules/AnalysisDisc.hpp"
#include "shammodels/sph/modules/AnalysisDustMass.hpp"
#include "shammodels/sph/modules/AnalysisEnergyKinetic.hpp"
#include "shammodels/sph/modules/AnalysisEnergyPotential.hpp"
#include "shammodels/sph/modules/AnalysisSodTube.hpp"
#include "shammodels/sph/modules/AnalysisTotalMomentum.hpp"
#include "shamphys/SodTube.hpp"
#include "shampylib/PatchDataToPy.hpp"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <memory>
#include <optional>
#include <random>
#include <utility>

template<class Tvec, template<class> class SPHKernel>
void add_instance(py::module &m, std::string name_config, std::string name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;

    using T = Model<Tvec, SPHKernel>;

    using TAnalysisSodTube = shammodels::sph::modules::AnalysisSodTube<Tvec, SPHKernel>;
    using TAnalysisDisc    = shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>;
    using TSPHSetup        = shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>;

    shamlog_debug_ln("[Py]", "registering class :", name_config, typeid(T).name());
    shamlog_debug_ln("[Py]", "registering class :", name_model, typeid(T).name());

    pybindings::add_solver_config_instance<Tvec, SPHKernel>(m, name_config);

    std::string sod_tube_analysis_name = name_model + "_AnalysisSodTube";
    py::class_<TAnalysisSodTube>(m, sod_tube_analysis_name.c_str())
        .def("compute_L2_dist", [](TAnalysisSodTube &self) -> std::tuple<Tscal, Tvec, Tscal> {
            auto ret = self.compute_L2_dist();
            return {ret.rho, ret.v, ret.P};
        });

    std::string disc_analysis_name = name_model + "_AnalysisDisc";
    py::class_<TAnalysisDisc>(m, disc_analysis_name.c_str())
        .def(
            "collect_data",
            [](TAnalysisDisc &self, Tscal Rmin, Tscal Rmax, u32 Nbin, ShamrockCtx &ctx) {
                auto anal = self.compute_analysis(Rmin, Rmax, Nbin, ctx);
                py::dict dic_out;

                auto radius  = anal.radius.copy_to_stdvec();
                auto counter = anal.counter.copy_to_stdvec();
                auto Sigma   = anal.Sigma.copy_to_stdvec();
                auto lx      = anal.lx.copy_to_stdvec();
                auto ly      = anal.ly.copy_to_stdvec();
                auto lz      = anal.lz.copy_to_stdvec();
                auto tilt    = anal.tilt.copy_to_stdvec();
                auto twist   = anal.twist.copy_to_stdvec();
                auto psi     = anal.psi.copy_to_stdvec();
                auto Hsq     = anal.Hsq.copy_to_stdvec();

                dic_out["radius"]  = radius;
                dic_out["counter"] = counter;
                dic_out["Sigma"]   = Sigma;
                dic_out["lx"]      = lx;
                dic_out["ly"]      = ly;
                dic_out["lz"]      = lz;
                dic_out["tilt"]    = tilt;
                dic_out["twist"]   = twist;
                dic_out["psi"]     = psi;
                dic_out["Hsq"]     = Hsq;

                return dic_out;
            });

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
            py::arg("init_h_factor") = modules::GeneratorLatticeHCP<Tvec>::default_init_h_factor,
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
            py::arg("init_h_factor") = modules::GeneratorLatticeFCC<Tvec>::default_init_h_factor,
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
                            "make_generator_disc_mc: either velocity_field or rot_profile must be "
                            "provided, you must provide one of them");
                    }

                    if (velocity_field && rot_profile) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "make_generator_disc_mc: either velocity_field or rot_profile must be "
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
                                    "make_generator_disc_mc: with the current EOS, cs_profile is "
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
                return self.make_modifier_warp_disc(parent, Rwarp, Hwarp, inclination, posangle);
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
                return self.make_modifier_custom_warp(parent, inc_profile, psi_profile, k_profile);
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

    pybindings::add_model_instance<Tvec, SPHKernel>(m, name_model);
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisBarycenter_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;

    using T = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisBarycenter<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisBarycenter<Tvec, SPHKernel>>(model);
        }))
        .def("get_barycenter", [](modules::AnalysisBarycenter<Tvec, SPHKernel> &self) {
            auto result = self.get_barycenter();
            return py::make_tuple(result.barycenter, result.mass_disc);
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisEnergyKinetic_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisEnergyKinetic<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisEnergyKinetic<Tvec, SPHKernel>>(model);
        }))
        .def("get_kinetic_energy", [](modules::AnalysisEnergyKinetic<Tvec, SPHKernel> &self) {
            return self.get_kinetic_energy();
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisEnergyPotential_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisEnergyPotential<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisEnergyPotential<Tvec, SPHKernel>>(model);
        }))
        .def("get_potential_energy", [](modules::AnalysisEnergyPotential<Tvec, SPHKernel> &self) {
            return self.get_potential_energy();
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisTotalMomentum_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisTotalMomentum<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisTotalMomentum<Tvec, SPHKernel>>(model);
        }))
        .def("get_total_momentum", [](modules::AnalysisTotalMomentum<Tvec, SPHKernel> &self) {
            return self.get_total_momentum();
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisAngularMomentum_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisAngularMomentum<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisAngularMomentum<Tvec, SPHKernel>>(model);
        }))
        .def("get_angular_momentum", [](modules::AnalysisAngularMomentum<Tvec, SPHKernel> &self) {
            return self.get_angular_momentum();
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisDustMass_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisDustMass<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisDustMass<Tvec, SPHKernel>>(model);
        }))
        .def("get_dust_mass", [](modules::AnalysisDustMass<Tvec, SPHKernel> &self) {
            return self.get_dust_mass();
        });
}

using namespace shammodels::sph;

template<class Analysis, typename Tvec, template<class> class SPHKernel>
auto analysis_impl(shammodels::sph::Model<Tvec, SPHKernel> &model) -> Analysis {
    return Analysis(model);
}

template<template<class, template<class> class> class Analysis>
void register_analysis_impl_for_each_kernel(py::module &msph, const char *name_class) {
    using namespace shammodels::sph;

    using SPHModel_f64_3_M4 = shammodels::sph::Model<f64_3, shammath::M4>;
    using SPHModel_f64_3_M6 = shammodels::sph::Model<f64_3, shammath::M6>;
    using SPHModel_f64_3_M8 = shammodels::sph::Model<f64_3, shammath::M8>;

    using SPHModel_f64_3_C2 = shammodels::sph::Model<f64_3, shammath::C2>;
    using SPHModel_f64_3_C4 = shammodels::sph::Model<f64_3, shammath::C4>;
    using SPHModel_f64_3_C6 = shammodels::sph::Model<f64_3, shammath::C6>;

    msph.def(
        name_class,
        [](SPHModel_f64_3_M4 &model) {
            return analysis_impl<Analysis<f64_3, shammath::M4>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_M6 &model) {
            return analysis_impl<Analysis<f64_3, shammath::M6>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_M8 &model) {
            return analysis_impl<Analysis<f64_3, shammath::M8>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_C2 &model) {
            return analysis_impl<Analysis<f64_3, shammath::C2>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_C4 &model) {
            return analysis_impl<Analysis<f64_3, shammath::C4>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_C6 &model) {
            return analysis_impl<Analysis<f64_3, shammath::C6>>(model);
        },
        py::kw_only(),
        py::arg("model"));
}

ON_PYTHON_INIT {
    auto &m = root_module;

    py::module msph = m.def_submodule("model_sph", "Shamrock sph solver");

    py::class_<shamrock::PatchDataLazyGetter>(m, "PatchDataLazyGetter")
        .def("__getitem__", &shamrock::PatchDataLazyGetter::get_item);

    py::class_<EvolveUntilResults>(m, "EvolveUntilResults")
        .def_readwrite("reach_target_time", &EvolveUntilResults::reach_target_time)
        .def_readwrite("reach_niter_max", &EvolveUntilResults::reach_niter_max)
        .def_readwrite("reach_max_walltime", &EvolveUntilResults::reach_max_walltime)
        .def_readwrite("iter_count", &EvolveUntilResults::iter_count)
        .def("__repr__", [](const EvolveUntilResults &self) {
            return sham::format(
                "EvolveUntilResults(reach_target_time={}, reach_niter_max={}, "
                "reach_max_walltime={}, iter_count={})",
                self.reach_target_time,
                self.reach_niter_max,
                self.reach_max_walltime,
                self.iter_count);
        });

    using namespace shammodels::sph;

    add_instance<f64_3, shammath::M4>(msph, "SPHModel_f64_3_M4_SolverConfig", "SPHModel_f64_3_M4");
    add_instance<f64_3, shammath::M6>(msph, "SPHModel_f64_3_M6_SolverConfig", "SPHModel_f64_3_M6");
    add_instance<f64_3, shammath::M8>(msph, "SPHModel_f64_3_M8_SolverConfig", "SPHModel_f64_3_M8");

    add_instance<f64_3, shammath::C2>(msph, "SPHModel_f64_3_C2_SolverConfig", "SPHModel_f64_3_C2");
    add_instance<f64_3, shammath::C4>(msph, "SPHModel_f64_3_C4_SolverConfig", "SPHModel_f64_3_C4");
    add_instance<f64_3, shammath::C6>(msph, "SPHModel_f64_3_C6_SolverConfig", "SPHModel_f64_3_C6");

    using VariantSPHModelBind = std::variant<
        std::unique_ptr<Model<f64_3, shammath::M4>>,
        std::unique_ptr<Model<f64_3, shammath::M6>>,
        std::unique_ptr<Model<f64_3, shammath::M8>>,
        std::unique_ptr<Model<f64_3, shammath::C2>>,
        std::unique_ptr<Model<f64_3, shammath::C4>>,
        std::unique_ptr<Model<f64_3, shammath::C6>>>;

    m.def(
        "get_Model_SPH",
        [](ShamrockCtx &ctx,
           const std::string &vector_type,
           const std::string &kernel) -> VariantSPHModelBind {
            VariantSPHModelBind ret;

            if (vector_type == "f64_3" && kernel == "M4") {
                ret = std::make_unique<Model<f64_3, shammath::M4>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "M6") {
                ret = std::make_unique<Model<f64_3, shammath::M6>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "M8") {
                ret = std::make_unique<Model<f64_3, shammath::M8>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "C2") {
                ret = std::make_unique<Model<f64_3, shammath::C2>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "C4") {
                ret = std::make_unique<Model<f64_3, shammath::C4>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "C6") {
                ret = std::make_unique<Model<f64_3, shammath::C6>>(ctx);
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "unknown combination of representation and kernel");
            }

            return ret;
        },
        py::kw_only(),
        py::arg("context"),
        py::arg("vector_type"),
        py::arg("sph_kernel"));

    py::class_<
        shammodels::sph::modules::ISPHSetupNode,
        std::shared_ptr<shammodels::sph::modules::ISPHSetupNode>>(msph, "ISPHSetupNode")
        .def("get_dot", [](std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> &self) {
            return self->get_dot();
        });

    py::class_<shammodels::sph::TimestepLog>(msph, "TimestepLog")
        .def(py::init<>())
        .def_readwrite("rank", &shammodels::sph::TimestepLog::rank)
        .def_readwrite("rate", &shammodels::sph::TimestepLog::rate)
        .def_readwrite("npart", &shammodels::sph::TimestepLog::npart)
        .def_readwrite("tcompute", &shammodels::sph::TimestepLog::tcompute)
        .def("rate_sum", &shammodels::sph::TimestepLog::rate_sum)
        .def("npart_sum", &shammodels::sph::TimestepLog::npart_sum);

    add_analysisBarycenter_instance<f64_3, shammath::M4>(msph, "AnalysisBarycenter_f64_3_M4");
    add_analysisBarycenter_instance<f64_3, shammath::M6>(msph, "AnalysisBarycenter_f64_3_M6");
    add_analysisBarycenter_instance<f64_3, shammath::M8>(msph, "AnalysisBarycenter_f64_3_M8");

    add_analysisBarycenter_instance<f64_3, shammath::C2>(msph, "AnalysisBarycenter_f64_3_C2");
    add_analysisBarycenter_instance<f64_3, shammath::C4>(msph, "AnalysisBarycenter_f64_3_C4");
    add_analysisBarycenter_instance<f64_3, shammath::C6>(msph, "AnalysisBarycenter_f64_3_C6");

    add_analysisEnergyKinetic_instance<f64_3, shammath::M4>(msph, "AnalysisEnergyKinetic_f64_3_M4");
    add_analysisEnergyKinetic_instance<f64_3, shammath::M6>(msph, "AnalysisEnergyKinetic_f64_3_M6");
    add_analysisEnergyKinetic_instance<f64_3, shammath::M8>(msph, "AnalysisEnergyKinetic_f64_3_M8");

    add_analysisEnergyKinetic_instance<f64_3, shammath::C2>(msph, "AnalysisEnergyKinetic_f64_3_C2");
    add_analysisEnergyKinetic_instance<f64_3, shammath::C4>(msph, "AnalysisEnergyKinetic_f64_3_C4");
    add_analysisEnergyKinetic_instance<f64_3, shammath::C6>(msph, "AnalysisEnergyKinetic_f64_3_C6");

    add_analysisEnergyPotential_instance<f64_3, shammath::M4>(
        msph, "AnalysisEnergyPotential_f64_3_M4");
    add_analysisEnergyPotential_instance<f64_3, shammath::M6>(
        msph, "AnalysisEnergyPotential_f64_3_M6");
    add_analysisEnergyPotential_instance<f64_3, shammath::M8>(
        msph, "AnalysisEnergyPotential_f64_3_M8");

    add_analysisEnergyPotential_instance<f64_3, shammath::C2>(
        msph, "AnalysisEnergyPotential_f64_3_C2");
    add_analysisEnergyPotential_instance<f64_3, shammath::C4>(
        msph, "AnalysisEnergyPotential_f64_3_C4");
    add_analysisEnergyPotential_instance<f64_3, shammath::C6>(
        msph, "AnalysisEnergyPotential_f64_3_C6");

    add_analysisTotalMomentum_instance<f64_3, shammath::M4>(msph, "AnalysisTotalMomentum_f64_3_M4");
    add_analysisTotalMomentum_instance<f64_3, shammath::M6>(msph, "AnalysisTotalMomentum_f64_3_M6");
    add_analysisTotalMomentum_instance<f64_3, shammath::M8>(msph, "AnalysisTotalMomentum_f64_3_M8");

    add_analysisTotalMomentum_instance<f64_3, shammath::C2>(msph, "AnalysisTotalMomentum_f64_3_C2");
    add_analysisTotalMomentum_instance<f64_3, shammath::C4>(msph, "AnalysisTotalMomentum_f64_3_C4");
    add_analysisTotalMomentum_instance<f64_3, shammath::C6>(msph, "AnalysisTotalMomentum_f64_3_C6");

    add_analysisAngularMomentum_instance<f64_3, shammath::M4>(
        msph, "AnalysisAngularMomentum_f64_3_M4");
    add_analysisAngularMomentum_instance<f64_3, shammath::M6>(
        msph, "AnalysisAngularMomentum_f64_3_M6");
    add_analysisAngularMomentum_instance<f64_3, shammath::M8>(
        msph, "AnalysisAngularMomentum_f64_3_M8");

    add_analysisAngularMomentum_instance<f64_3, shammath::C2>(
        msph, "AnalysisAngularMomentum_f64_3_C2");
    add_analysisAngularMomentum_instance<f64_3, shammath::C4>(
        msph, "AnalysisAngularMomentum_f64_3_C4");
    add_analysisAngularMomentum_instance<f64_3, shammath::C6>(
        msph, "AnalysisAngularMomentum_f64_3_C6");

    register_analysis_impl_for_each_kernel<modules::AnalysisBarycenter>(msph, "analysisBarycenter");
    register_analysis_impl_for_each_kernel<modules::AnalysisEnergyKinetic>(
        msph, "analysisEnergyKinetic");
    register_analysis_impl_for_each_kernel<modules::AnalysisEnergyPotential>(
        msph, "analysisEnergyPotential");
    register_analysis_impl_for_each_kernel<modules::AnalysisTotalMomentum>(
        msph, "analysisTotalMomentum");
    register_analysis_impl_for_each_kernel<modules::AnalysisAngularMomentum>(
        msph, "analysisAngularMomentum");

    add_analysisDustMass_instance<f64_3, shammath::M4>(msph, "AnalysisDustMass_f64_3_M4");
    add_analysisDustMass_instance<f64_3, shammath::M6>(msph, "AnalysisDustMass_f64_3_M6");
    add_analysisDustMass_instance<f64_3, shammath::M8>(msph, "AnalysisDustMass_f64_3_M8");

    add_analysisDustMass_instance<f64_3, shammath::C2>(msph, "AnalysisDustMass_f64_3_C2");
    add_analysisDustMass_instance<f64_3, shammath::C4>(msph, "AnalysisDustMass_f64_3_C4");
    add_analysisDustMass_instance<f64_3, shammath::C6>(msph, "AnalysisDustMass_f64_3_C6");

    register_analysis_impl_for_each_kernel<modules::AnalysisDustMass>(msph, "analysisDustMass");
}
