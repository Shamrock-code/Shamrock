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
 * @file ForceKernelBase.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Template Method pattern for GSPH force computation
 *
 * Encapsulates the common infrastructure for neighbor-loop force computation:
 * - Field index acquisition
 * - Buffer access management
 * - SYCL queue and event handling
 * - Neighbor cache iteration
 *
 * Derived classes (per physics domain) implement only the physics-specific
 * inner kernel logic, eliminating the massive code duplication in UpdateDerivs.
 *
 * The GSPH method originated from:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 */

#include "shambase/StackEntry.hpp"
#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/FieldNames.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels::gsph::core {

    using shamrock::PatchScheduler;
    using shamrock::patch::Patch;
    using shamrock::patch::PatchDataLayer;
    using shamrock::patch::PatchDataLayerLayout;

    /**
     * @brief Common particle state accessible in force kernels
     *
     * Contains the fields that are common across all physics domains.
     * Physics-specific fields are handled by derived ForceKernel classes.
     */
    template<class Tvec>
    struct CommonParticleState {
        using Tscal = shambase::VecComponent<Tvec>;

        Tvec xyz;         ///< Position (lab-frame)
        Tvec vxyz;        ///< Velocity (lab-frame)
        Tscal hpart;      ///< Smoothing length
        Tscal omega;      ///< Grad-h correction factor Ω
        Tscal density; ///< Lab-frame baryon density N (kernel summation)
        Tscal pressure;   ///< Pressure (rest-frame)
        Tscal cs;         ///< Sound speed (rest-frame)
    };

    /**
     * @brief Base class for GSPH force kernel implementations
     *
     * Uses the Template Method pattern to define the algorithm skeleton for
     * force computation while allowing physics-specific customization.
     *
     * Common operations (setup, buffer acquisition, event handling) are
     * implemented once here. Derived classes provide:
     * - Additional field setup (setup_physics_fields)
     * - Additional buffer acquisition (acquire_physics_buffers)
     * - The core pairwise force computation (compute_pairwise_force)
     * - Result accumulation (accumulate_result)
     * - Event completion (complete_physics_events)
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class ForceKernelBase {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Kernel             = SPHKernel<Tscal>;
        using Storage            = SolverStorage<Tvec, u32>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        ForceKernelBase(PatchScheduler &sched, Storage &storage)
            : sched_(sched), storage_(storage) {}

        virtual ~ForceKernelBase() = default;

        /**
         * @brief Execute the force computation (Template Method)
         *
         * This is the main entry point. It defines the algorithm skeleton:
         * 1. Setup common field indices
         * 2. Setup physics-specific field indices (hook)
         * 3. For each patch:
         *    a. Acquire common buffers
         *    b. Acquire physics-specific buffers (hook)
         *    c. Submit SYCL kernel with neighbor iteration
         *    d. Complete events
         */
        void execute() {
            StackEntry stack_loc{};

            setup_common_fields();
            setup_physics_fields(); // Hook for derived classes

            auto &merged_xyzh = storage_.merged_xyzh.get();
            auto &mpdats      = storage_.merged_patchdata_ghost.get();

            sched_.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

                // Acquire common buffers
                CommonBuffers bufs = acquire_common_buffers(cur_p, pdat, mpdat);

                // Hook for physics-specific buffers
                acquire_physics_buffers(cur_p, pdat, mpdat);

                // Get neighbor cache
                tree::ObjectCache &pcache
                    = shambase::get_check_ref(storage_.neigh_cache).get_cache(cur_p.id_patch);

                // Submit kernel
                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                sham::EventList depends_list;

                auto e = submit_force_kernel(q, depends_list, pdat, pcache, bufs);

                // Complete common events
                complete_common_events(e, bufs);

                // Hook for physics-specific event completion
                complete_physics_events(e);

                sham::EventList resulting_events;
                resulting_events.add_event(e);
                pcache.complete_event_state(resulting_events);
            });
        }

        protected:
        // ════════════════════════════════════════════════════════════════════════
        // Hooks for derived classes (Template Method pattern)
        // ════════════════════════════════════════════════════════════════════════

        /**
         * @brief Setup physics-specific field indices
         *
         * Called once before iterating over patches. Derived classes should
         * store any additional field indices needed (e.g., SR: dS, de indices).
         */
        virtual void setup_physics_fields() = 0;

        /**
         * @brief Acquire physics-specific buffers for a patch
         *
         * Called for each patch before kernel submission. Derived classes should
         * acquire any additional buffers and store accessors.
         *
         * @param patch Current patch
         * @param pdat Patch data layer
         * @param mpdat Merged patch data (includes ghosts)
         */
        virtual void acquire_physics_buffers(
            Patch patch, PatchDataLayer &pdat, PatchDataLayer &mpdat)
            = 0;

        /**
         * @brief Complete physics-specific event states
         *
         * Called after kernel completion. Derived classes should complete
         * event states for any additional buffers they acquired.
         *
         * @param e Completed SYCL event
         */
        virtual void complete_physics_events(sycl::event e) = 0;

        /**
         * @brief Get particle mass from configuration
         * @return Particle mass
         */
        virtual Tscal get_particle_mass() const = 0;

        /**
         * @brief Get EOS gamma from configuration
         * @return Adiabatic index γ
         */
        virtual Tscal get_eos_gamma() const = 0;

        // ════════════════════════════════════════════════════════════════════════
        // Accessors for derived classes
        // ════════════════════════════════════════════════════════════════════════

        PatchScheduler &scheduler() { return sched_; }
        Storage &storage() { return storage_; }
        PatchDataLayerLayout &pdl() { return sched_.pdl(); }

        // Common field indices (set during setup_common_fields)
        u32 idx_xyz() const { return ixyz_; }
        u32 idx_vxyz() const { return ivxyz_; }
        u32 idx_hpart() const { return ihpart_; }
        u32 idx_hpart_interf() const { return ihpart_interf_; }
        u32 idx_vxyz_interf() const { return ivxyz_interf_; }
        u32 idx_omega_interf() const { return iomega_interf_; }
        u32 idx_density_interf() const { return idensity_interf_; }

        private:
        /**
         * @brief Common buffers used in force computation
         */
        struct CommonBuffers {
            sham::DeviceBuffer<Tvec> *buf_xyz;
            sham::DeviceBuffer<Tvec> *buf_vxyz;
            sham::DeviceBuffer<Tscal> *buf_hpart;
            sham::DeviceBuffer<Tscal> *buf_omega;
            sham::DeviceBuffer<Tscal> *buf_density;
            sham::DeviceBuffer<Tscal> *buf_pressure;
            sham::DeviceBuffer<Tscal> *buf_cs;
        };

        /// Index for position field in merged_xyzh structure
        static constexpr u32 kMergedPosIdx = 0;

        void setup_common_fields() {
            using namespace shammodels::gsph;
            PatchDataLayerLayout &pdl = sched_.pdl();

            ixyz_   = pdl.get_field_idx<Tvec>(fields::XYZ);
            ivxyz_  = pdl.get_field_idx<Tvec>(fields::VXYZ);
            ihpart_ = pdl.get_field_idx<Tscal>(fields::HPART);

            // Ghost layout indices
            PatchDataLayerLayout &ghost_layout
                = shambase::get_check_ref(storage_.ghost_layout.get());
            ihpart_interf_   = ghost_layout.get_field_idx<Tscal>(fields::HPART);
            ivxyz_interf_    = ghost_layout.get_field_idx<Tvec>(fields::VXYZ);
            iomega_interf_   = ghost_layout.get_field_idx<Tscal>(fields::OMEGA);
            // Note: Density field name is mode-specific, derived classes should
            // override get_density_field_name() if they use a different name
            idensity_interf_ = ghost_layout.get_field_idx<Tscal>(get_density_field_name());
        }

        /**
         * @brief Get the density field name for this physics mode
         *
         * Each physics mode must implement this to return its specific density
         * field name:
         * - Newtonian: computed_fields::DENSITY (mass density ρ)
         * - SR: fields::N_LABFRAME (lab-frame baryon density N)
         *
         * @return Field name string
         */
        virtual const char *get_density_field_name() const = 0;

        CommonBuffers acquire_common_buffers(
            Patch cur_p, PatchDataLayer &pdat, PatchDataLayer &mpdat) {
            auto &merged_xyzh = storage_.merged_xyzh.get();

            CommonBuffers bufs;
            bufs.buf_xyz = &merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(
                kMergedPosIdx);
            bufs.buf_vxyz       = &mpdat.get_field_buf_ref<Tvec>(ivxyz_interf_);
            bufs.buf_hpart      = &mpdat.get_field_buf_ref<Tscal>(ihpart_interf_);
            bufs.buf_omega      = &mpdat.get_field_buf_ref<Tscal>(iomega_interf_);
            bufs.buf_density    = &mpdat.get_field_buf_ref<Tscal>(idensity_interf_);

            auto &pressure_field = shambase::get_check_ref(storage_.pressure);
            auto &cs_field       = shambase::get_check_ref(storage_.soundspeed);
            bufs.buf_pressure    = &pressure_field.get_field(cur_p.id_patch).get_buf();
            bufs.buf_cs          = &cs_field.get_field(cur_p.id_patch).get_buf();

            return bufs;
        }

        /**
         * @brief Submit the force computation kernel (pure virtual)
         *
         * Derived classes must implement the actual force computation kernel.
         * This method is called by execute() for each patch.
         *
         * @param q SYCL device queue
         * @param depends_list Event dependencies
         * @param pdat Patch data layer
         * @param pcache Neighbor cache
         * @param bufs Common buffers
         * @return SYCL event for completion tracking
         */
        virtual sycl::event submit_force_kernel(
            sham::DeviceQueue &q,
            sham::EventList &depends_list,
            PatchDataLayer &pdat,
            tree::ObjectCache &pcache,
            CommonBuffers &bufs)
            = 0;

        void complete_common_events(sycl::event e, CommonBuffers &bufs) {
            bufs.buf_xyz->complete_event_state(e);
            bufs.buf_vxyz->complete_event_state(e);
            bufs.buf_hpart->complete_event_state(e);
            bufs.buf_omega->complete_event_state(e);
            bufs.buf_density->complete_event_state(e);
            bufs.buf_pressure->complete_event_state(e);
            bufs.buf_cs->complete_event_state(e);
        }

        PatchScheduler &sched_;
        Storage &storage_;

        // Common field indices
        u32 ixyz_               = 0;
        u32 ivxyz_              = 0;
        u32 ihpart_             = 0;
        u32 ihpart_interf_      = 0;
        u32 ivxyz_interf_       = 0;
        u32 iomega_interf_      = 0;
        u32 idensity_interf_ = 0;
    };

} // namespace shammodels::gsph::core
