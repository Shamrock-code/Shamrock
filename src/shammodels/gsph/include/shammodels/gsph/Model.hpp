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
 * @file Model.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief GSPH Model class - high-level interface for GSPH simulations
 *
 * The GSPH method originated from:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 *
 * This implementation follows:
 * - Cha, S.-H. & Whitworth, A.P. (2003) "Implementations and tests of
 *   Godunov-type particle hydrodynamics"
 */

#include "shambase/constants.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shambackends/BufferMirror.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/setup/generators.hpp"
#include "shammodels/gsph/Solver.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/io/ShamrockDump.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include <pybind11/functional.h>
#include <stdexcept>
#include <vector>

namespace shammodels::gsph {

    /**
     * @brief The GSPH Model class
     *
     * Provides a high-level interface for setting up and running GSPH simulations.
     * The GSPH method uses Riemann solvers at particle interfaces instead of
     * artificial viscosity, giving sharper shock resolution.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel Kernel type (e.g., M4, M6, C2, C4, C6 for Wendland)
     */
    template<class Tvec, template<class> class SPHKernel>
    class Model {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Solver       = Solver<Tvec, SPHKernel>;
        using SolverConfig = typename Solver::Config;

        ShamrockCtx &ctx;
        Solver solver;

        Model(ShamrockCtx &ctx) : ctx(ctx), solver(ctx) {};

        ////////////////////////////////////////////////////////////////////////////////////////////
        // Setup functions
        ////////////////////////////////////////////////////////////////////////////////////////////

        void init_scheduler(u32 crit_split, u32 crit_merge);

        template<std::enable_if_t<dim == 3, int> = 0>
        inline Tvec get_box_dim_fcc_3d(Tscal dr, u32 xcnt, u32 ycnt, u32 zcnt) {
            return generic::setup::generators::get_box_dim(dr, xcnt, ycnt, zcnt);
        }

        inline void set_cfl_cour(Tscal cfl_cour) {
            solver.solver_config.cfl_config.cfl_cour = cfl_cour;
        }

        inline void set_cfl_force(Tscal cfl_force) {
            solver.solver_config.cfl_config.cfl_force = cfl_force;
        }

        inline void set_particle_mass(Tscal gpart_mass) {
            solver.solver_config.gpart_mass = gpart_mass;
        }

        inline Tscal get_particle_mass() { return solver.solver_config.gpart_mass; }

        inline void resize_simulation_box(std::pair<Tvec, Tvec> box) {
            ctx.set_coord_domain_bound({box.first, box.second});
        }

        void do_vtk_dump(std::string filename, bool add_patch_world_id) {
            solver.vtk_do_dump(filename, add_patch_world_id);
        }

        u64 get_total_part_count();

        f64 total_mass_to_part_mass(f64 totmass);

        std::pair<Tvec, Tvec> get_ideal_fcc_box(Tscal dr, std::pair<Tvec, Tvec> box);
        std::pair<Tvec, Tvec> get_ideal_hcp_box(Tscal dr, std::pair<Tvec, Tvec> box);

        Tscal get_hfact() { return Kernel::hfactd; }

        Tscal rho_h(Tscal h) {
            return shamrock::sph::rho_h(solver.solver_config.gpart_mass, h, Kernel::hfactd);
        }

        void add_cube_fcc_3d(Tscal dr, std::pair<Tvec, Tvec> _box);
        void add_cube_hcp_3d(Tscal dr, std::pair<Tvec, Tvec> _box);

        ////////////////////////////////////////////////////////////////////////////////////////////
        // Field manipulation
        ////////////////////////////////////////////////////////////////////////////////////////////

        /**
         * @brief Apply a position-dependent function to initialize a field
         *
         * Sets field values by evaluating a function at each particle position.
         * Useful for setting up spatially-varying initial conditions.
         *
         * @tparam T Field type (e.g., Tscal for density, Tvec for velocity)
         * @param field_name Name of the field to modify (e.g., "uint", "vxyz")
         * @param pos_to_val Function mapping position to field value
         *
         * Example:
         * @code
         * // Set velocity as a function of position
         * model.apply_field_from_position<Tvec>("vxyz", [](Tvec pos) {
         *     return Tvec{pos[0], 0.0, 0.0};  // Linear velocity profile
         * });
         * @endcode
         */
        template<class T>
        inline void apply_field_from_position(
            std::string field_name, const std::function<T(Tvec)> pos_to_val) {

            StackEntry stack_loc{};
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            sched.patch_data.for_each_patchdata(
                [&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
                    PatchDataField<Tvec> &xyz
                        = pdat.template get_field<Tvec>(sched.pdl().get_field_idx<Tvec>("xyz"));

                    PatchDataField<T> &f
                        = pdat.template get_field<T>(sched.pdl().get_field_idx<T>(field_name));

                    if (f.get_nvar() != 1) {
                        shambase::throw_unimplemented();
                    }

                    {
                        auto &buf = f.get_buf();
                        auto acc  = buf.copy_to_stdvec();

                        auto &buf_xyz = xyz.get_buf();
                        auto acc_xyz  = buf_xyz.copy_to_stdvec();

                        for (u32 i = 0; i < f.get_obj_cnt(); i++) {
                            Tvec r = acc_xyz[i];
                            acc[i] = pos_to_val(r);
                        }

                        buf.copy_from_stdvec(acc);
                        buf_xyz.copy_from_stdvec(acc_xyz);
                    }
                });
        }

        /**
         * @brief Set field value for particles within a box region
         *
         * Sets the specified field to a constant value for all particles
         * whose positions fall within the given axis-aligned box.
         * Useful for setting up discontinuous initial conditions (e.g., Sod shock tube).
         *
         * @tparam T Field type (e.g., Tscal for scalars, Tvec for vectors)
         * @param field_name Name of the field to modify (e.g., "uint", "vxyz")
         * @param val Value to set for particles in the region
         * @param box Bounding box as (min_corner, max_corner)
         * @param ivar Variable index for multi-variable fields (default: 0)
         *
         * Example:
         * @code
         * // Sod shock tube: set left state internal energy
         * model.set_field_in_box("uint", u_left, {box_min, interface_pos});
         * // Set right state
         * model.set_field_in_box("uint", u_right, {interface_pos, box_max});
         * @endcode
         */
        template<class T>
        inline void set_field_in_box(
            std::string field_name, T val, std::pair<Tvec, Tvec> box, u32 ivar = 0) {
            StackEntry stack_loc{};
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            sched.patch_data.for_each_patchdata(
                [&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
                    PatchDataField<Tvec> &xyz
                        = pdat.template get_field<Tvec>(sched.pdl().get_field_idx<Tvec>("xyz"));

                    PatchDataField<T> &f
                        = pdat.template get_field<T>(sched.pdl().get_field_idx<T>(field_name));

                    u32 nvar = f.get_nvar();

                    // Validate ivar parameter to prevent out-of-bounds access
                    if (ivar >= nvar) {
                        shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                            "set_field_in_box: ivar ({}) >= f.get_nvar ({}) for field {}",
                            ivar,
                            nvar,
                            field_name));
                    }

                    {
                        auto acc     = f.get_buf().template mirror_to<sham::host>();
                        auto acc_xyz = xyz.get_buf().template mirror_to<sham::host>();

                        for (u32 i = 0; i < f.get_obj_cnt(); i++) {
                            Tvec r = acc_xyz[i];

                            if (BBAA::is_coord_in_range(r, std::get<0>(box), std::get<1>(box))) {
                                acc[i * nvar + ivar] = val;
                            }
                        }
                    }
                });
        }

        /**
         * @brief Set field value for particles within a spherical region
         *
         * Sets the specified field to a constant value for all particles
         * whose positions fall within the given sphere.
         * Useful for setting up point-source initial conditions (e.g., Sedov blast).
         *
         * @tparam T Field type (must be single-variable, e.g., Tscal)
         * @param field_name Name of the field to modify (e.g., "uint")
         * @param val Value to set for particles in the region
         * @param center Center of the sphere
         * @param radius Radius of the sphere
         *
         * Example:
         * @code
         * // Sedov blast: inject energy in central sphere
         * Tscal blast_energy_per_particle = E_blast / n_particles_in_sphere;
         * model.set_field_in_sphere("uint", blast_energy_per_particle, origin, r_blast);
         * @endcode
         */
        template<class T>
        inline void set_field_in_sphere(std::string field_name, T val, Tvec center, Tscal radius) {
            StackEntry stack_loc{};
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            sched.patch_data.for_each_patchdata(
                [&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
                    PatchDataField<Tvec> &xyz
                        = pdat.template get_field<Tvec>(sched.pdl().get_field_idx<Tvec>("xyz"));

                    PatchDataField<T> &f
                        = pdat.template get_field<T>(sched.pdl().get_field_idx<T>(field_name));

                    if (f.get_nvar() != 1) {
                        shambase::throw_unimplemented();
                    }

                    Tscal r2 = radius * radius;
                    {
                        auto acc     = f.get_buf().template mirror_to<sham::host>();
                        auto acc_xyz = xyz.get_buf().template mirror_to<sham::host>();

                        for (u32 i = 0; i < f.get_obj_cnt(); i++) {
                            Tvec dr = acc_xyz[i] - center;

                            if (sycl::dot(dr, dr) < r2) {
                                acc[i] = val;
                            }
                        }
                    }
                });
        }

        template<class T>
        inline T get_sum(std::string name) {
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            T sum                 = shambase::VectorProperties<T>::get_zero();

            StackEntry stack_loc{};
            sched.patch_data.for_each_patchdata(
                [&](u64 patch_id, shamrock::patch::PatchDataLayer &pdat) {
                    PatchDataField<T> &xyz
                        = pdat.template get_field<T>(sched.pdl().get_field_idx<T>(name));

                    sum += xyz.compute_sum();
                });

            return shamalgs::collective::allreduce_sum(sum);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        // Solver configuration
        ////////////////////////////////////////////////////////////////////////////////////////////

        inline SolverConfig gen_default_config() {
            SolverConfig cfg;
            cfg.set_riemann_iterative();              // Default to iterative Riemann solver
            cfg.set_reconstruct_piecewise_constant(); // Default to 1st order (piecewise constant)
            cfg.set_eos_adiabatic(Tscal{1.4});
            cfg.set_boundary_periodic();
            return cfg;
        }

        inline void set_solver_config(SolverConfig cfg) {
            if (ctx.is_scheduler_initialized()) {
                shambase::throw_with_loc<std::runtime_error>(
                    "Cannot change solver config after scheduler is initialized");
            }
            cfg.check_config();
            solver.solver_config = cfg;
        }

        inline f64 solver_logs_last_rate() { return solver.solve_logs.get_last_rate(); }
        inline u64 solver_logs_last_obj_count() { return solver.solve_logs.get_last_obj_count(); }

        ////////////////////////////////////////////////////////////////////////////////////////////
        // I/O (uses shared ShamrockDump mechanism like SPH)
        ////////////////////////////////////////////////////////////////////////////////////////////

        inline void load_from_dump(std::string fname) {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln("GSPH", "Loading state from dump", fname);
            }

            std::string metadata_user{};
            shamrock::load_shamrock_dump(fname, metadata_user, ctx);

            nlohmann::json j = nlohmann::json::parse(metadata_user);
            j.at("solver_config").get_to(solver.solver_config);

            solver.init_ghost_layout();
            solver.init_solver_graph();

            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            sched.owned_patch_id  = sched.patch_list.build_local();
            sched.patch_list.build_local_idx_map();
            sched.patch_list.build_global_idx_map();
            sched.update_local_load_value([&](shamrock::patch::Patch p) {
                return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
            });
        }

        inline void dump(std::string fname) {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln("GSPH", "Dumping state to", fname);
            }

            solver.update_sync_load_values();

            nlohmann::json metadata;
            metadata["solver_config"] = solver.solver_config;

            shamrock::write_shamrock_dump(
                fname, metadata.dump(4), shambase::get_check_ref(ctx.sched));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        // Simulation control
        ////////////////////////////////////////////////////////////////////////////////////////////

        TimestepLog timestep() { return solver.evolve_once(); }

        inline void evolve_once() {
            solver.evolve_once();
            solver.print_timestep_logs();
        }

        inline bool evolve_until(Tscal target_time, i32 niter_max = -1) {
            return solver.evolve_until(target_time, niter_max);
        }
    };

} // namespace shammodels::gsph
