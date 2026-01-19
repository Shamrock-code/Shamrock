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
 * @file simulation_domain.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include <memory>
#include <optional>
#include <vector>

enum BCType { Free, Periodic, PeriodicShearing, Fixed, Ghost, FixedGradient, AntiPeriodic };

template<class flt>
class SimulationDomain {
    public:
    BCType boundary_type;

    using vec     = sycl::vec<flt, 3>;
    using vec_box = std::tuple<vec, vec>;

    ALignedAxisBoundingBox<flt> box_bc;

    std::optional<u32_3> periodic_search_min_vec;
    std::optional<u32_3> periodic_search_max_vec;

    SimulationDomain(const BCType &boundary_type, vec min, vec max)
        : box_bc(ALignedAxisBoundingBox<flt>(min, max)), boundary_type(boundary_type) {
        check_boundary();
    }

    inline bool has_outdomain_object() {
        switch (boundary_type) {
        case Periodic        : return true; break;
        case PeriodicShearing: return true; break;
        default              :;
        }
        return false;
    }

    /*
     * describe the schearing cd in the following way
     * std::tuple<vec_box, u32_3, u32, vec> equivalent to [shearing vec, field id (flt type x 3),
     * vec] applied like this : field[fid] = field[fid] + vec*dot(pvec,svec); svec is the shearing
     * vector (argument) pvec is the vector that discribe the periodicity
     */
    // std::vector<std::tuple<u32_3, u32, vec>> shear_cd;

    inline void check_boundary() {
        if (boundary_type == PeriodicShearing) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "[SimulationDomain] Boundary CD : Shearing periodic mode not implemented");
        }

        if (boundary_type == Fixed) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "[SimulationDomain] Boundary CD : Dirichelt mode not implemented");
        }

        if (boundary_type == Ghost) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "[SimulationDomain] Boundary CD : Ghost mode not implemented");
        }

        if (boundary_type == FixedGradient) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "[SimulationDomain] Boundary CD : FixedGradient mode not implemented");
        }

        if (boundary_type == AntiPeriodic) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "[SimulationDomain] Boundary CD : AntiPeriodic mode not implemented");
        }
    }

    inline void set_box(const vec_box &b) {
        box_bc = ALignedAxisBoundingBox<flt>(std::get<0>(b), std::get<1>(b));
    }

    inline vec get_periodicity_vector() const {
        vec pvec;

        if (boundary_type == Periodic || boundary_type == PeriodicShearing) {
            pvec = box_bc.get_size();
        } else if (boundary_type == Free) {
            pvec = vec{0, 0, 0};
        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "[SimulationDomain] Can not set box size with free boundary conditions");
        }

        return pvec;
    }

    inline void set_periodic_search_range(u32_3 min, u32_3 max) {
        if (!(has_outdomain_object())) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "[SimulationDomain] Can not set periodic search range without periodic bc");
        }
        periodic_search_min_vec = min;
        periodic_search_max_vec = max;
    }
};
