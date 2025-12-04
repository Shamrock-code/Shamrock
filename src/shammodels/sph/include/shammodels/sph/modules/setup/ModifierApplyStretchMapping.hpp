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

#include "shambase/logs/loglevels.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/CoordinateTransformation.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class ModifierApplyStretchMapping : public ISPHSetupNode {
        using Tscal               = shambase::VecComponent<Tvec>;
        static constexpr u32 dim  = shambase::VectorProperties<Tvec>::dimension;
        static constexpr Tscal pi = shambase::constants::pi<Tscal>;

        using Config = SolverConfig<Tvec, SPHKernel>;
        using Kernel = SPHKernel<Tscal>;

        static constexpr Tscal tol     = 1e-9;
        static constexpr u32 maxits    = 300;
        static constexpr u32 maxits_nr = 30;

        struct Smap_inputdata {
            Tvec center;
            std::vector<std::function<Tscal(Tscal)>> rhoprofiles;
            std::string system;
            std::vector<std::string> axes;

            std::vector<std::function<Tscal(Tscal)>> rhodSs;
            std::vector<std::function<Tscal(Tvec)>> a_from_poss;
            std::vector<std::function<Tvec(Tscal, Tvec)>> a_to_poss;
            std::vector<Tscal> integral_profiles;
            std::vector<Tscal> steps;
            Tvec boxmin;
            Tvec boxmax;
            std::vector<Tscal> ximins;
            std::vector<Tscal> ximaxs;
        };

        ShamrockCtx &context;
        Config &solver_config;

        SetupNodePtr parent;

        public:
        static constexpr u32 ngrid = 2048;
        Tscal mpart                = solver_config.gpart_mass;
        Tscal hfact                = Kernel::hfactd;

        Smap_inputdata smap_inputdata;

        public:
        ModifierApplyStretchMapping(
            ShamrockCtx &context,
            Config &solver_config,
            SetupNodePtr parent,
            std::vector<std::function<Tscal(Tscal)>> rhoprofiles,
            std::string system,
            std::vector<std::string> axes)
            : context(context), solver_config(solver_config), parent(parent) {

            // TODO: assert that rhoprofiles and axes have same length.
            // TODO: Optimisations avec std:make ?
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

            Tvec boxmin = {-1, -1, -1}; // TODO Il va falloir recover ça
            Tvec boxmax = {1, 1, 1};    // TODO Il va falloir recover ça
            Tvec center = (boxmin + boxmax) / 2;

            for (size_t k = 0; k < rhoprofiles.size(); ++k) {
                auto rhoprofile = rhoprofiles[k];
                auto axis       = axes[k];
                size_t axisnb   = hashAxis(axis, system);

                std::function<Tscal(Tscal)> S;
                std::function<Tscal(Tvec)>
                    a_from_pos; // from a cartesian position get the coordinate
                                // to stretch in the appropriate system
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
                            return sycl::atan2(pos.y(), pos.x()); // phi
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
                            return sycl::atan2(pos.y(), pos.x());
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

                integral_profile = shammath::integ_trapezoidal(thisxmin, thisxmax, step, rhodS);

                smap_inputdata.rhoprofiles.push_back(rhoprofile);
                smap_inputdata.rhodSs.push_back(rhodS);
                smap_inputdata.a_from_poss.push_back(a_from_pos);
                smap_inputdata.a_to_poss.push_back(a_to_pos);
                smap_inputdata.integral_profiles.push_back(integral_profile);
                smap_inputdata.steps.push_back(step);
                smap_inputdata.ximins.push_back(thisxmin);
                smap_inputdata.ximaxs.push_back(thisxmax);
            }
            smap_inputdata.center = center;
            smap_inputdata.system = system;
            smap_inputdata.axes   = axes;
            smap_inputdata.boxmin = boxmin;
            smap_inputdata.boxmax = boxmax;
            shamlog_debug_ln("ModifierApplyStretchMapping", "Stretch mapping...")
        }
        /**
         * @brief stretch a single coordinate of a particle position
         *
         * @tparam Tscal a,
         * @tparam std::function<Tscal(Tscal)> const &rhoprofile,
         * @tparam std::function<Tscal(Tscal)> const &rhodS,
         * @tparam Tscal integral_profile,
         * @tparam Tscal amin,
         * @tparam Tscal amax,
         * @tparam Tvec center,
         * @tparam Tscal step
         */
        static Tscal stretchcoord(
            Tscal a,
            std::function<Tscal(Tscal)> const &rhoprofile,
            std::function<Tscal(Tscal)> const &rhodS,
            Tscal integral_profile,
            Tscal amin,
            Tscal amax,
            Tvec center,
            Tscal step) {

            u32 its       = 0;
            Tscal initpos = a;
            Tscal newr    = a; // initial guess
            Tscal prevr   = newr;

            Tscal initrelatpos
                = (sycl::pown(initpos, 3) - sycl::pown(amin, 3))
                  / (sycl::pown(amax, 3) - sycl::pown(amin, 3)); // TODO Only works in spherical
            //                         case!!
            Tscal newrelatpos
                = shammath::integ_trapezoidal(amin, newr, step, rhodS) / integral_profile;
            Tscal func       = newrelatpos - initrelatpos;
            Tscal xminbisect = amin;
            Tscal xmaxbisect = amax;
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
                if ((newr > amax) || (newr < amin) || (its > maxits_nr)) {
                    bisect = true;
                    newr   = 0.5 * (xminbisect + xmaxbisect);
                }
                newrelatpos
                    = shammath::integ_trapezoidal(amin, newr, step, rhodS) / integral_profile;
                func  = newrelatpos - initrelatpos;
                prevr = newr;
            }
            return newr;
        }

        static Tvec stretchpart(Tvec pos, Smap_inputdata smap_inputdata) {
            Tvec center = smap_inputdata.center;
            Tvec cpos   = pos - center;
            if (sycl::length(cpos) < 1e-12) {
                return center;
            }
            for (size_t i = 0; i < smap_inputdata.rhoprofiles.size(); ++i) {
                auto &rhoprofile       = smap_inputdata.rhoprofiles[i];
                auto &rhodS            = smap_inputdata.rhodSs[i];
                auto &ximax            = smap_inputdata.ximaxs[i];
                auto &ximin            = smap_inputdata.ximins[i];
                auto &a_from_pos       = smap_inputdata.a_from_poss[i];
                auto &a_to_pos         = smap_inputdata.a_to_poss[i];
                auto &integral_profile = smap_inputdata.integral_profiles[i];
                auto &step             = smap_inputdata.steps[i];
                Tscal a                = a_from_pos(cpos);

                Tscal new_a = stretchcoord(
                    a, rhoprofile, rhodS, integral_profile, ximin, ximax, center, step);
                cpos = a_to_pos(new_a, cpos);
            }
            return center + cpos;
        }

        static Tscal h_rho_stretched(
            Tvec pos, const Smap_inputdata &smap_inputdata, Tscal mpart, Tscal hfact) {
            Tscal rhoa = 1.;
            for (size_t i = 0; i < smap_inputdata.integral_profiles.size(); i++) {
                Tscal integral_profile = smap_inputdata.integral_profiles[i];
                auto &a_from_pos       = smap_inputdata.a_from_poss[i];
                auto &rhoprofile       = smap_inputdata.rhoprofiles[i];

                rhoa *= rhoprofile(a_from_pos(pos - smap_inputdata.center)) / integral_profile;
            }

            Tscal h = shamrock::sph::h_rho(
                1.,
                rhoa,
                hfact); // to be divided by number of particles ^(1/3) (Thus we don't need the mass)
            return h;
        }

        bool is_done() { return parent->is_done(); }

        shamrock::patch::PatchDataLayer next_n(u32 nmax);

        std::string get_name() { return "ModifierApplyStretchMapping"; }
        ISPHSetupNode_Dot get_dot_subgraph() {
            return ISPHSetupNode_Dot{get_name(), 2, {parent->get_dot_subgraph()}};
        }
    }; // namespace shammodels::sph::modules
} // namespace shammodels::sph::modules
