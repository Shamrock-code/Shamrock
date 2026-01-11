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
 * @file ComputeEOSNode.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solvergraph node for equation of state computation
 *
 * Computes thermodynamic quantities (pressure, sound speed) from
 * density and internal energy using the configured EOS.
 *
 * For Newtonian: P = (\gamma - 1) \rho u, c_s = \sqrt{\gamma P / \rho}
 * For SR: Extended to include enthalpy and relativistic sound speed
 */

#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::gsph::solvergraph {

    /**
     * @brief Compute thermodynamic quantities from EOS
     *
     * This node wraps EOS evaluation for all particles.
     * Different EOS types (ideal gas, polytrope, tabulated) are handled
     * via the EOSType template parameter.
     *
     * Inputs:
     * - density: SPH-summed density
     * - uint: Specific internal energy
     * - part_counts: Particle counts per patch
     *
     * Outputs:
     * - pressure: Thermodynamic pressure
     * - soundspeed: Local sound speed
     *
     * @tparam Tvec Vector type
     * @tparam EOSType Equation of state type (must have P(rho, u), cs(rho, u))
     */
    template<class Tvec, class EOSType>
    class ComputeEOSNode : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        /// EOS instance
        EOSType eos;

        public:
        /**
         * @brief Edge container for type-safe access
         */
        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shamrock::solvergraph::Field<Tscal> &density;
            const shamrock::solvergraph::Field<Tscal> &uint;
            shamrock::solvergraph::Field<Tscal> &pressure;
            shamrock::solvergraph::Field<Tscal> &soundspeed;
        };

        /**
         * @brief Construct EOS computation node
         *
         * @param eos Equation of state instance
         */
        explicit ComputeEOSNode(EOSType eos) : eos(std::move(eos)) {}

        /**
         * @brief Set input/output edges
         */
        void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> uint,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> pressure,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> soundspeed) {
            __internal_set_ro_edges({part_counts, density, uint});
            __internal_set_rw_edges({pressure, soundspeed});
        }

        /**
         * @brief Get typed edge references
         */
        Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(2),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(1)};
        }

        /**
         * @brief Get the EOS instance
         */
        const EOSType &get_eos() const { return eos; }

        std::string _impl_get_label() const override { return "ComputeEOS"; }

        std::string _impl_get_tex() const override { return "P, c_s \\leftarrow \\rho, u"; }

        protected:
        /**
         * @brief Execute the computation
         *
         * Evaluates EOS for all particles to compute pressure and sound speed.
         */
        void _impl_evaluate_internal() override;
    };

    // =========================================================================
    // Extended EOS nodes for different physics (STUBS)
    // =========================================================================

    /**
     * @brief SR EOS node (STUB)
     *
     * For special relativity, EOS evaluation also computes:
     * - Specific enthalpy h = 1 + u + P/\rho
     * - Relativistic sound speed
     *
     * @tparam Tvec Vector type
     * @tparam EOSType Equation of state type
     */
    template<class Tvec, class EOSType>
    class ComputeEOSSRNode : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        EOSType eos;

        public:
        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shamrock::solvergraph::Field<Tscal> &density;
            const shamrock::solvergraph::Field<Tscal> &uint;
            shamrock::solvergraph::Field<Tscal> &pressure;
            shamrock::solvergraph::Field<Tscal> &soundspeed;
            shamrock::solvergraph::Field<Tscal> &enthalpy;
        };

        explicit ComputeEOSSRNode(EOSType eos) : eos(std::move(eos)) {}

        void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> uint,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> pressure,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> soundspeed,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> enthalpy) {
            __internal_set_ro_edges({part_counts, density, uint});
            __internal_set_rw_edges({pressure, soundspeed, enthalpy});
        }

        Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(2),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(1),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(2)};
        }

        std::string _impl_get_label() const override { return "ComputeEOS_SR"; }

        std::string _impl_get_tex() const override { return "P, c_s, h \\leftarrow \\rho, u"; }

        protected:
        void _impl_evaluate_internal() override {
            // STUB: SR EOS computation not yet implemented
        }
    };

} // namespace shammodels::gsph::solvergraph
