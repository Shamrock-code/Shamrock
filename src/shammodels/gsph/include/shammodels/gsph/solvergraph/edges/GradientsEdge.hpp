// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GradientsEdge.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solvergraph edge for MUSCL gradient fields
 *
 * Bundles all gradient fields used in MUSCL reconstruction:
 * - grad_density: \nabla \rho
 * - grad_pressure: \nabla P
 * - grad_vx, grad_vy, grad_vz: \nabla v
 */

#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"

namespace shammodels::gsph::solvergraph {

    /**
     * @brief Edge bundling MUSCL gradient fields
     *
     * Groups all gradient fields together for cleaner storage management.
     * Only allocated when MUSCL reconstruction is enabled.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     */
    template<class Tvec>
    class GradientsEdge : public shamrock::solvergraph::IEdgeNamed {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        /// Gradient fields (each stores vector gradient per particle)
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_density;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_pressure;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vx;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vy;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vz;

        /**
         * @brief Construct gradient edge
         *
         * @param name Edge name
         * @param tex_symbol TeX symbol for visualization
         */
        GradientsEdge(std::string name, std::string tex_symbol)
            : IEdgeNamed(std::move(name), std::move(tex_symbol)) {
            initialize_fields();
        }

        /**
         * @brief Ensure all gradient fields are sized correctly
         *
         * @param sizes Particle counts per patch
         */
        void ensure_sizes(const shambase::DistributedData<u32> &sizes) {
            grad_density->ensure_sizes(sizes);
            grad_pressure->ensure_sizes(sizes);
            grad_vx->ensure_sizes(sizes);
            grad_vy->ensure_sizes(sizes);
            grad_vz->ensure_sizes(sizes);
        }

        /**
         * @brief Free all allocations
         */
        void free_alloc() override {
            if (grad_density) {
                grad_density->free_alloc();
            }
            if (grad_pressure) {
                grad_pressure->free_alloc();
            }
            if (grad_vx) {
                grad_vx->free_alloc();
            }
            if (grad_vy) {
                grad_vy->free_alloc();
            }
            if (grad_vz) {
                grad_vz->free_alloc();
            }
        }

        private:
        void initialize_fields() {
            grad_density = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
                1, "grad_density", "\\nabla\\rho");
            grad_pressure = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
                1, "grad_pressure", "\\nabla P");
            grad_vx
                = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "grad_vx", "\\nabla v_x");
            grad_vy
                = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "grad_vy", "\\nabla v_y");
            grad_vz
                = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "grad_vz", "\\nabla v_z");
        }
    };

} // namespace shammodels::gsph::solvergraph
