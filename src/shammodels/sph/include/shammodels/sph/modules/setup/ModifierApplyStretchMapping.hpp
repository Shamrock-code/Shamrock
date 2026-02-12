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
#include <cstddef>

template<typename T, typename arr_t>
std::array<size_t, 2> get_closest_range(const arr_t &arr, const T &val, size_t size) {
    size_t low = 0, high = size - 1;

    if (val < arr[low]) {
        return {low, low};
    }

    if (val > arr[high]) {
        return {high, high};
    }

    while (high - low > 1) {

        size_t mid = (low + high) / 2;

        if (arr[mid] < val) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return {low, high};
}

template<typename T, typename arr_t>
T linear_interpolate(const arr_t &arr_x, const arr_t &arr_y, size_t arr_size, const T &x) {

    auto closest_range = get_closest_range(arr_x, x, arr_size);
    size_t left_idx    = closest_range[0];
    size_t right_idx   = closest_range[1];

    if (left_idx == right_idx) {
        return arr_y[left_idx];
    }

    T x0 = arr_x[left_idx];
    T x1 = arr_x[right_idx];
    T y0 = arr_y[left_idx];
    T y1 = arr_y[right_idx];

    if (x1 == x0) {
        return std::numeric_limits<T>::signaling_NaN();
    }

    T interpolated_y = y0 + (x - x0) / (x1 - x0) * (y1 - y0);

    return interpolated_y;
} // from SedovTaylor

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

        shammath::AABB<Tvec> box;

        Tscal mtot;

        struct TabulatedDensity {
            std::vector<Tscal> x;
            std::vector<Tscal> rho;
        };

        struct Smap_inputdata {
            Tvec center;
            std::function<Tscal(Tscal)> rhoprofile;
            std::string system;
            std::string axis;

            std::function<Tscal(Tscal)> rhodS;
            std::function<Tscal(Tvec)> a_from_pos;
            std::function<Tvec(Tscal, Tvec)> a_to_pos;
            Tscal integral_profile;
            Tscal step;
            Tvec boxmin;
            Tvec boxmax;
            Tscal ximin;
            Tscal ximax;
        };

        ShamrockCtx &context;
        Config &solver_config;

        SetupNodePtr parent;

        public:
        static constexpr u32 ngrid = 2048;
        Tscal hfact                = Kernel::hfactd;

        Smap_inputdata smap_inputdata;

        public:
        ModifierApplyStretchMapping(
            ShamrockCtx &context,
            Config &solver_config,
            SetupNodePtr parent,
            std::vector<Tscal> tabrho,
            std::vector<Tscal> tabx,
            std::string system,
            std::string axis,
            std::pair<Tvec, Tvec> box,
            Tscal mtot)
            : context(context), solver_config(solver_config), parent(parent), box(box), mtot(mtot) {

            TabulatedDensity tabul = {tabx, tabrho};
            shamlog_debug_ln("ModifierApplyStretchMapping", "tabul", tabul.x, tabul.rho);

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

            Tvec boxmin = box.first;  // TODO Recover it from parent?
            Tvec boxmax = box.second; // TODO Recover it from parent?
            Tvec center = (boxmin + boxmax) / 2;

            // auto rhoprofile = rhoprofiles[k];
            // std::function<Tscal(Tscal)> rhoprofile;
            auto rhoprofile = [tabul](Tscal r) -> Tscal {
                return linear_interpolate(tabul.x, tabul.rho, tabul.x.size(), r);
            };
            // auto rhoprofile = [](Tscal r) -> Tscal {
            //         if (r < 1e-12) {
            //             return 1;
            //         }
            //         Tscal r_adim = r * sycl::sqrt(2 * pi); // n=1
            //         return sycl::sin(r_adim) / r_adim;};

            size_t axisnb = hashAxis(axis, system);

            std::function<Tscal(Tscal)> S;
            std::function<Tscal(Tvec)> a_from_pos; // from a cartesian position get the coordinate
                                                   // to stretch in the appropriate system
            std::function<Tvec(Tscal, Tvec)>
                a_to_pos; // from the appropriate system, convert stretched coordinate into
                          // cartesian position
            Tscal integral_profile;

            Tscal lx   = sham::abs(boxmin.x() - boxmax.x());
            Tscal ly   = sham::abs(boxmin.y() - boxmax.y());
            Tscal lz   = sham::abs(boxmin.z() - boxmax.z());
            Tscal lmin = sycl::min(lx, ly);
            lmin       = sycl::min(lmin, lz);

            if (system == "spherical") {
                x0min   = 0; // There's a particle at r=0 because the HCP is centered on (0,0,0)
                x0max   = lmin / 2.;
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
                x0max   = lmin / 2.;
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
                    new_pos[i] = pos[i] * (transfo)[i][axisnb](new_yk) / (transfo)[i][axisnb](yk);
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

            smap_inputdata.rhoprofile       = rhoprofile;
            smap_inputdata.rhodS            = rhodS;
            smap_inputdata.a_from_pos       = a_from_pos;
            smap_inputdata.a_to_pos         = a_to_pos;
            smap_inputdata.integral_profile = integral_profile;
            smap_inputdata.step             = step;
            smap_inputdata.ximin            = thisxmin;
            smap_inputdata.ximax            = thisxmax;

            smap_inputdata.center = center;
            smap_inputdata.system = system;
            smap_inputdata.axis   = axis;
            smap_inputdata.boxmin = boxmin;
            smap_inputdata.boxmax = boxmax;
            shambase::println("Stretch mapping...");
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

            auto &rhoprofile       = smap_inputdata.rhoprofile;
            auto &rhodS            = smap_inputdata.rhodS;
            auto &ximax            = smap_inputdata.ximax;
            auto &ximin            = smap_inputdata.ximin;
            auto &a_from_pos       = smap_inputdata.a_from_pos;
            auto &a_to_pos         = smap_inputdata.a_to_pos;
            auto &integral_profile = smap_inputdata.integral_profile;
            auto &step             = smap_inputdata.step;
            Tscal a                = a_from_pos(cpos);

            Tscal new_a
                = stretchcoord(a, rhoprofile, rhodS, integral_profile, ximin, ximax, center, step);
            cpos = a_to_pos(new_a, cpos);

            return center + cpos;
        }

        static Tscal h_rho_stretched(Tvec pos, const Smap_inputdata &smap_inputdata, Tscal hfact) {
            Tscal rhoa = 1.;

            Tscal integral_profile = smap_inputdata.integral_profile;
            auto &a_from_pos       = smap_inputdata.a_from_pos;
            auto &rhoprofile       = smap_inputdata.rhoprofile;

            rhoa *= rhoprofile(a_from_pos(pos - smap_inputdata.center)) / integral_profile;

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
