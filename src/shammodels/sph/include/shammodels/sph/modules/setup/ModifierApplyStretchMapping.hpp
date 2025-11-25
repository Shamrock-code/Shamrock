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
 * @file ModifierApplyStretchMapping.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 *
 */

#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/CoordinateTransformation.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class ModifierApplyStretchMapping : public ISPHSetupNode {
        using Tscal               = shambase::VecComponent<Tvec>;
        static constexpr u32 dim  = shambase::VectorProperties<Tvec>::dimension;
        static constexpr Tscal pi = shambase::constants::pi<Tscal>;

        using Config = SolverConfig<Tvec, SPHKernel>;
        using Kernel = SPHKernel<Tscal>;

        static constexpr Tscal tol     = 1e-9;
        static constexpr u32 maxits    = 200;
        static constexpr u32 maxits_nr = 20;

        ShamrockCtx &context;
        Config &solver_config;

        SetupNodePtr parent;

        public:
        std::vector<std::function<Tscal(Tscal)>> rhoprofiles;
        std::string system;
        std::vector<std::string> axes;

        Tvec center;
        std::vector<std::function<Tscal(Tscal)>> rhodSs;
        std::vector<std::function<Tscal(Tvec)>> a_from_poss;
        std::vector<std::function<Tvec(Tscal, Tvec)>> a_to_poss;
        std::vector<Tscal> integral_profiles;
        std::vector<Tscal> steps;
        Tvec boxmin = {-1, -1, -1}; // Il va falloir recover ça
        Tvec boxmax = {1, 1, 1};    // Il va falloir recover ça
        std::vector<Tscal> ximins;  // will be sent to lattce
        std::vector<Tscal> ximaxs;  // will be sent to lattce
        static constexpr u32 ngrid = 2048;

        Tscal mpart = solver_config.gpart_mass;
        Tscal hfact = Kernel::hfactd;

        public:
        ModifierApplyStretchMapping(
            ShamrockCtx &context,
            Config &solver_config,
            SetupNodePtr parent,
            std::vector<std::function<Tscal(Tscal)>> rhoprofiles,
            std::string system,
            std::vector<std::string> axes)
            : context(context), solver_config(solver_config), parent(parent),
              rhoprofiles(rhoprofiles), system(system), axes(axes) {
            Tscal x0min;
            Tscal x0max;
            Tscal x1min;
            Tscal x1max;
            Tscal x2min;
            Tscal x2max;
            CartToCart<Tvec> tocart;
            CartToSpherical<Tvec> tospherical;
            CartToCylindrical<Tvec> tocylindrical;
            std::array<std::array<std::function<Tscal(Tscal)>, 3>, 3> transfo;
            center = (boxmin + boxmax) / 2;

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

                Tscal lx = sham::abs(boxmin.x() - boxmax.x());
                Tscal ly = sham::abs(boxmin.y() - boxmax.y());
                Tscal lz = sham::abs(boxmin.z() - boxmax.z());

                if (system == "spherical") {
                    x0min   = 0; // There's a particle at r=0 because the HCP is centered on (0,0,0)
                    x0max   = sham::abs((boxmax - boxmin).x()) / 2.;
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
                    x0max   = sham::abs((boxmax - boxmin).x()) / 2.;
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
                    shamlog_error("ModifierApplyStretchMapping", "Ah non hein");
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
        }
        /**
         * @brief stretch a single coordinate of a particle position
         *
         * @tparam Tscal a,
         * @tparam std::function<Tscal(Tscal)> const &rhoprofile,
         * @tparam std::function<Tscal(Tscal)> const &rhodS,
         * @tparam Tscal integral_profile,
         * @tparam Tscal rmin,
         * @tparam Tscal rmax,
         * @tparam Tvec center,
         * @tparam Tscal step
         */
        static Tscal stretchcoord(
            Tscal a,
            std::function<Tscal(Tscal)> const &rhoprofile,
            std::function<Tscal(Tscal)> const &rhodS,
            Tscal integral_profile,
            Tscal rmin,
            Tscal rmax,
            Tvec center,
            Tscal step) {

            u32 its       = 0;
            Tscal initpos = a;
            Tscal newr    = a;
            Tscal prevr   = a;

            Tscal initrelatpos = (sycl::pown(initpos, 3) - sycl::pown(rmin, 3))
                                 / (sycl::pown(rmax, 3) - sycl::pown(rmin, 3));
            Tscal newrelatpos
                = shammath::integ_riemann_sum(rmin, newr, step, rhodS) / integral_profile;
            Tscal func       = newrelatpos - initrelatpos;
            Tscal xminbisect = rmin;
            Tscal xmaxbisect = rmax;
            Tscal dfunc      = 0;
            Tscal dx         = 0;
            bool bisect      = false;
            while ((sham::abs(func) > tol) && (its < maxits)) {
                its++;
                if (bisect) {
                    if (func > 0.) {
                        xmaxbisect = newr;
                    } else {
                        xminbisect = newr;
                    }
                    newr = 0.5 * (xminbisect + xmaxbisect);
                } else {
                    dfunc = rhodS(newr) / integral_profile;
                    dx    = func / dfunc;
                    newr  = newr - dx;
                }

                if (sham::abs(newr) < 0.8 * sham::abs(prevr)) {
                    newr = 0.8 * prevr;
                } else if (sham::abs(newr) > 1.2 * sham::abs(prevr)) {
                    newr = 1.2 * prevr;
                }

                // NR iteration
                if ((newr > rmax) || (newr < rmin) || (its > maxits_nr)) {
                    bisect = true;
                    newr   = 0.5 * (xminbisect + xmaxbisect);
                }
                newrelatpos
                    = shammath::integ_riemann_sum(rmin, newr, step, rhodS) / integral_profile;
                func  = newrelatpos - initrelatpos;
                prevr = newr;
            }
            return newr;
        }

        static Tvec stretchpart(
            Tvec pos,
            std::vector<std::function<Tscal(Tscal)>> rhoprofiles,
            std::vector<std::function<Tscal(Tscal)>> rhodSs,
            std::vector<std::function<Tscal(Tvec)>> a_from_poss,
            std::vector<std::function<Tvec(Tscal, Tvec)>> a_to_poss,
            std::vector<Tscal> integral_profiles,
            std::vector<Tscal> ximins,
            std::vector<Tscal> ximaxs,
            Tvec center,
            std::vector<Tscal> steps) {
            for (size_t i = 0; i < rhoprofiles.size(); ++i) {
                auto &rhoprofile       = rhoprofiles[i];
                auto &rhodS            = rhodSs[i];
                auto &ximax            = ximaxs[i];
                auto &ximin            = ximins[i];
                auto &a_from_pos       = a_from_poss[i];
                auto &a_to_pos         = a_to_poss[i];
                auto &integral_profile = integral_profiles[i];
                auto &step             = steps[i];
                Tscal a                = a_from_pos(pos);
                // a_from_pos =  sycl::length e.g
                // or a_from_pos = pos_a.get("x")
                Tscal new_a = stretchcoord(
                    a, rhoprofile, rhodS, integral_profile, ximin, ximax, center, step);
                pos = a_to_pos(new_a, pos);
            }
            return pos - center;
        }

        static Tscal h_rho_stretched(
            Tvec pos,
            std::vector<std::function<Tscal(Tscal)>> rhoprofiles,
            std::vector<std::function<Tscal(Tvec)>> a_from_poss,
            std::vector<Tscal> integral_profiles,
            Tscal mpart,
            Tscal hfact) {
            Tscal rhoa = 1.;
            for (size_t i = 0; i < integral_profiles.size(); i++) {
                Tscal integral_profile = integral_profiles[i];
                auto &a_from_pos       = a_from_poss[i];
                auto &rhoprofile       = rhoprofiles[i];

                rhoa *= rhoprofile(a_from_pos(pos)) / integral_profile;
            }

            Tscal h = shamrock::sph::h_rho(
                mpart, rhoa, hfact); // to be multiplied by number of particles
            return h;
        }

        static Tscal test(Tscal x) { return x + 1; }
        // static Tvec testvec(Tvec x) { return x * 2.; }
        // static auto testvec = [](Tvec x) -> Tvec {
        //     return x * 2.;
        // };
        static Tscal test2(Tscal x, Tscal y) { return x + y; }

        bool is_done() { return parent->is_done(); }

        shamrock::patch::PatchDataLayer next_n(u32 nmax);
        static Tvec testvec(Tvec x);

        std::string get_name() { return "ModifierApplyStretchMapping"; }
        ISPHSetupNode_Dot get_dot_subgraph() {
            return ISPHSetupNode_Dot{get_name(), 2, {parent->get_dot_subgraph()}};
        }
    };
} // namespace shammodels::sph::modules
