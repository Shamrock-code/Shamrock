// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GeneratorLatticeHCP_smap.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 * Moved to a Modifier. To be removed eventually
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammath/AABB.hpp"
#include "shammath/DiscontinuousIterator.hpp"
#include "shammath/crystalLattice_smap.hpp"
#include "shammath/integrator.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/CoordinateTransformation.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <shambackends/sycl.hpp>
#include <cstddef>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class GeneratorLatticeHCP_smap : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using LatticeHCP_smap    = shammath::LatticeHCP_smap<Tvec>;
        using LatticeIterDiscont_smap =
            typename shammath::LatticeHCP_smap<Tvec>::IteratorDiscontinuous;
        using Config = SolverConfig<Tvec, SPHKernel>;
        using Kernel = SPHKernel<Tscal>;

        static constexpr Tscal pi = shambase::constants::pi<Tscal>;

        ShamrockCtx &context;
        Config &solver_config;
        shammath::AABB<Tvec> box;

        LatticeHCP_smap lattice;
        LatticeIterDiscont_smap discont_iterator;

        static auto init_lattice(
            Tscal dr,
            std::pair<Tvec, Tvec> box,
            std::vector<std::function<Tscal(Tscal)>> rhoprofiles,
            std::string system,
            std::vector<std::string> axes) {
            static constexpr u32 ngrid = 2048;

            auto [idxs_min, idxs_max]
                = LatticeHCP_smap ::get_box_index_bounds(dr, box.first, box.second);

            Tvec center = (box.first + box.second) / 2;

            std::vector<std::function<Tscal(Tscal)>> rhodSs;
            std::vector<std::function<Tscal(Tvec)>> a_from_poss;
            std::vector<std::function<Tvec(Tscal, Tvec)>> a_to_poss;
            std::vector<Tscal> integral_profiles;
            std::vector<Tscal> steps;

            Tscal x0min;
            Tscal x0max;
            Tscal x1min;
            Tscal x1max;
            Tscal x2min;
            Tscal x2max;

            std::vector<Tscal> ximins; // will be sent to lattce
            std::vector<Tscal> ximaxs; // will be sent to lattce

            CartToCart<Tvec> tocart;
            CartToSpherical<Tvec> tospherical;
            CartToCylindrical<Tvec> tocylindrical;
            std::array<std::array<std::function<Tscal(Tscal)>, 3>, 3> transfo;

            for (size_t k = 0; k < rhoprofiles.size(); ++k) {
                auto rhoprofile = rhoprofiles[k];
                auto axis       = axes[k];
                size_t axisnb   = hashAxis(axis, system);

                std::function<Tscal(Tscal)> S;
                std::function<Tscal(Tvec)>
                    a_from_pos; // from a cartesian position get the coordinate to stretch in the
                                // appropriate system
                std::function<Tvec(Tscal, Tvec)>
                    a_to_pos; // from the appropriate system, convert stretched coordinate into
                              // cartesian position
                Tscal integral_profile;

                Tscal lx = sham::abs(box.first.x() - box.second.x());
                Tscal ly = sham::abs(box.first.y() - box.second.y());
                Tscal lz = sham::abs(box.first.z() - box.second.z());

                if (system == "spherical") {
                    x0min   = 0; // There's a particle at r=0 because the HCP is centered on (0,0,0)
                    x0max   = sham::abs((box.second - box.first).x()) / 2.;
                    x1min   = 0;
                    x1max   = pi;
                    x2min   = 0;
                    x2max   = 2 * pi;
                    transfo = tospherical.matrix;

                    switch (axisnb) {
                    case 0:
                        S = [](Tscal r) -> Tscal {
                            return 4 * pi * sycl::pown(r, 2);
                        };
                        a_from_pos = [](Tvec pos) -> Tscal {
                            return sycl::length(pos); // r
                        };
                        break;
                    case 1:
                        S = [x0max](Tscal theta) -> Tscal {
                            return 2 * pi * sycl::pown(x0max, 3) * sycl::sin(theta) / 3;
                        };
                        a_from_pos = [](Tvec pos) -> Tscal {
                            return sycl::acos(pos.z() / sycl::length(pos)); // theta
                        };
                        break;
                    case 2:
                        S = [x0max](Tscal phi) -> Tscal {
                            return 2 * pi * sycl::pown(x0max, 3) / 3;
                        };
                        a_from_pos = [](Tvec pos) -> Tscal {
                            return sycl::atan(pos.y() / pos.x()); // phi
                        };
                        break;
                    }
                } else if (system == "cylindrical") {
                    x0min   = 0;
                    x0max   = sham::abs((box.second - box.first).x()) / 2.;
                    x1min   = 0;
                    x1max   = 2. * pi;
                    x2min   = -lz / 2.;
                    x2max   = lz / 2.;
                    transfo = tocylindrical.matrix;
                    switch (axisnb) {
                    case 0:
                        S = [lz](Tscal r) -> Tscal {
                            return 4 * pi * r * lz;
                        };
                        a_from_pos = [](Tvec pos) -> Tscal {
                            return sycl::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
                        };
                        break;
                    case 1:
                        S = [x0max, lz](Tscal theta) -> Tscal {
                            return sycl::pown(x0max, 2) * lz / 2;
                        };
                        a_from_pos = [](Tvec pos) -> Tscal {
                            return sycl::atan(pos.y() / pos.x());
                        };
                        break;
                    case 2:
                        S = [x0max, lz](Tscal z) -> Tscal {
                            return pi * sycl::pown(x0max, 2) * lz;
                        };
                        a_from_pos = [](Tvec pos) -> Tscal {
                            return pos.z();
                        };
                        break;
                    }
                } else if (system == "cart") {
                    x0min   = -lx / 2.;
                    x0max   = lx / 2.;
                    x1min   = -ly / 2.;
                    x1max   = ly / 2.;
                    x2min   = -lz / 2.;
                    x2max   = lz / 2.;
                    transfo = tocart.matrix;
                    switch (axisnb) {
                    case 0:
                        S = [ly, lz](Tscal x) -> Tscal {
                            return ly * lz;
                        };
                        a_from_pos = [](Tvec pos) -> Tscal {
                            return pos.x();
                        };
                        break;
                    case 1:
                        S = [lx, lz](Tscal y) -> Tscal {
                            return lx * lz;
                        };
                        a_from_pos = [](Tvec pos) -> Tscal {
                            return pos.y();
                        };
                        break;
                    case 2:
                        S = [ly, lz](Tscal z) -> Tscal {
                            return ly * lz;
                        };
                        a_from_pos = [](Tvec pos) -> Tscal {
                            return pos.z();
                        };
                        break;
                    }
                } else {
                    shamlog_error("GeneratorLatticeHCP_smap", "Ah non hein");
                }

                a_to_pos = [a_from_pos, transfo, axisnb](Tscal new_yk, Tvec pos) -> Tvec {
                    Tvec new_pos;
                    Tscal yk = a_from_pos(pos);
                    for (size_t i = 0; i < pos.size(); ++i) {
                        new_pos[i]
                            = pos[i] * (transfo)[i][axisnb](new_yk) / (transfo)[i][axisnb](yk);
                    }
                    return new_pos;
                };

                auto rhodS = [rhoprofile, S](Tscal xi) -> Tscal {
                    return rhoprofile(xi) * S(xi);
                };
                std::vector<Tscal> ximin
                    = {x0min, x1min, x2min}; // temporary, will not be sent to lattice
                std::vector<Tscal> ximax
                    = {x0max, x1max, x2max}; // temporary, will not be sent to lattice

                Tscal thisxmin = ximin[axisnb]; // integral boundaries for later, along axis k
                                                // (that will be stretched)
                Tscal thisxmax = ximax[axisnb]; // integral boundaries for later, along axis k
                                                // (that will be stretched)
                Tscal step = (thisxmax - thisxmin) / ngrid;

                integral_profile = shammath::integ_riemann_sum(thisxmin, thisxmax, step, rhodS);

                rhodSs.push_back(rhodS);
                a_from_poss.push_back(a_from_pos);
                a_to_poss.push_back(a_to_pos);
                integral_profiles.push_back(integral_profile);
                steps.push_back(step);
                ximins.push_back(thisxmin);
                ximaxs.push_back(thisxmax);
            }

            return LatticeHCP_smap(
                dr,
                idxs_min,
                idxs_max,
                rhoprofiles,
                rhodSs,
                a_from_poss,
                a_to_poss,
                integral_profiles,
                ximins,
                ximaxs,
                center,
                steps);
        }

        public:
        GeneratorLatticeHCP_smap(
            ShamrockCtx &context,
            Config &solver_config,
            Tscal dr,
            std::pair<Tvec, Tvec> box,
            std::vector<std::function<Tscal(Tscal)>> rhoprofiles,
            std::string system,
            std::vector<std::string> axes)
            : context(context), solver_config(solver_config), box(box),
              lattice(init_lattice(dr, box, rhoprofiles, system, axes)),
              discont_iterator(lattice.get_IteratorDiscontinuous()) {};

        bool is_done() { return discont_iterator.is_done(); }

        shamrock::patch::PatchDataLayer next_n(u32 nmax) {
            StackEntry stack_loc{};

            using namespace shamrock::patch;
            PatchScheduler &sched = shambase::get_check_ref(context.sched);

            auto has_pdat = [&]() {
                bool ret = false;
                sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
                    ret = true;
                });
                return ret;
            };

            std::vector<Tvec> pos_data;
            std::vector<Tscal> h_data;
            int nbdebug = 0;

            // Fill pos_data and h_data if the scheduler has some patchdata in this rank
            if (!is_done()) {
                u64 loc_gen_count = (has_pdat()) ? nmax : 0;

                auto gen_info = shamalgs::collective::fetch_view(loc_gen_count);

                u64 skip_start = gen_info.head_offset;
                u64 gen_cnt    = loc_gen_count;
                u64 skip_end   = gen_info.total_byte_count - loc_gen_count - gen_info.head_offset;

                shamlog_debug_ln(
                    "GeneratorLatticeHCP_smap ",
                    "generate : ",
                    skip_start,
                    gen_cnt,
                    skip_end,
                    "total",
                    skip_start + gen_cnt + skip_end);

                discont_iterator.skip(skip_start);
                auto tmp = discont_iterator.next_n(gen_cnt);
                discont_iterator.skip(skip_end);

                for (Tvec r : tmp) {
                    nbdebug++;
                    if (Patch::is_in_patch_converted(r, box.lower, box.upper)) {
                        pos_data.push_back(r);
                    }
                    // if (Patch::is_in_patch_converted (r, box.upper)) {
                    //     pos_data.push_back(r);
                    // }
                }
                shamlog_debug_ln(
                    "GeneratorLatticeHCP_smap",
                    "from:",
                    pos_data.size(),
                    "to",
                    nbdebug,
                    "particles");
            }

            Tscal mpart = solver_config.gpart_mass;
            u64 npart   = pos_data.size();
            shamlog_debug_ln(
                "GeneratorLatticeHCP_smap",
                "pos_data.size:",
                pos_data.size(),
                "box:",
                box.lower,
                box.upper);

            Tscal totmass = mpart * npart;
            Tscal hfact   = Kernel::hfactd;

            for (Tvec &pos : pos_data) {

                Tscal rhoa = totmass;
                for (size_t i = 0; i < lattice.integral_profiles.size(); i++) {
                    Tscal integral_profile = lattice.integral_profiles[i];
                    auto &a_from_pos       = lattice.a_from_poss[i];
                    auto &rhoprofile       = lattice.rhoprofiles[i];

                    rhoa *= rhoprofile(a_from_pos(pos)) / integral_profile;
                }

                Tscal h = shamrock::sph::h_rho(mpart, rhoa, hfact);
                h_data.push_back(h);
            }

            // Make a patchdata from pos_data
            PatchDataLayer tmp(sched.get_layout_ptr());
            if (!pos_data.empty()) {
                tmp.resize(pos_data.size());
                tmp.fields_raz();

                {
                    u32 len = pos_data.size();
                    PatchDataField<Tvec> &f
                        = tmp.get_field<Tvec>(sched.pdl().get_field_idx<Tvec>("xyz"));
                    // sycl::buffer<Tvec> buf(pos_data.data(), len);
                    f.override(pos_data, len);
                }

                {
                    u32 len = pos_data.size();
                    PatchDataField<Tscal> &f
                        = tmp.get_field<Tscal>(sched.pdl().get_field_idx<Tscal>("hpart"));
                    f.override(h_data, len);
                    // sycl::buffer<Tscal> buf(h_data.data(), len);
                    // f.override(buf, len);
                }
            }
            return tmp;
        }

        std::string get_name() { return "GeneratorLatticeHCP_smap "; }
        ISPHSetupNode_Dot get_dot_subgraph() { return ISPHSetupNode_Dot{get_name(), 0, {}}; }
    };

} // namespace shammodels::sph::modules
