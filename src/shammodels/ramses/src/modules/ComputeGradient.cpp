// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeGradient.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammath/riemann.hpp"
#include "shammath/slopeLimiter.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/ComputeGradient.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <utility>

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraphLinkiterator;

namespace {

    template<class T>
    inline T slope_function_van_leer_f_form(T sL, T sR) {
        T st = sL + sR;

        auto vanleer = [](T f) {
            return 4. * f * (1. - f);
        };

        auto slopelim = [&](T f) {
            if constexpr (std::is_same_v<T, f64_3>) {
                f.x() = (f.x() >= 0 && f.x() <= 1) ? f.x() : 0;
                f.y() = (f.y() >= 0 && f.y() <= 1) ? f.y() : 0;
                f.z() = (f.z() >= 0 && f.z() <= 1) ? f.z() : 0;
            } else {
                f = (f >= 0 && f <= 1) ? f : 0;
            }
            return vanleer(f);
        };

        return slopelim(sL / st) * st * 0.5;
    }

    template<class T>
    inline T slope_function_van_leer_symetric(T sL, T sR) {

        if constexpr (std::is_same_v<T, f64_3>) {
            return {
                shammath::van_leer_slope_symetric(sL[0], sR[0]),
                shammath::van_leer_slope_symetric(sL[1], sR[1]),
                shammath::van_leer_slope_symetric(sL[2], sR[2])};
        } else {
            return shammath::van_leer_slope_symetric(sL, sR);
        }
    }

    template<class T>
    inline T slope_function_van_leer_standard(T sL, T sR) {

        if constexpr (std::is_same_v<T, f64_3>) {
            return {
                shammath::van_leer_slope(sL[0], sR[0]),
                shammath::van_leer_slope(sL[1], sR[1]),
                shammath::van_leer_slope(sL[2], sR[2])};
        } else {
            return shammath::van_leer_slope(sL, sR);
        }
    }

    template<class T>
    inline T slope_function_minmod(T sL, T sR) {

        if constexpr (std::is_same_v<T, f64_3>) {
            return {
                shammath::minmod(sL[0], sR[0]),
                shammath::minmod(sL[1], sR[1]),
                shammath::minmod(sL[2], sR[2])};
        } else {
            return shammath::minmod(sL, sR);
        }
    }

    using SlopeMode = shammodels::basegodunov::SlopeMode;

    template<class T, SlopeMode mode>
    inline T slope_function(T sL, T sR) {
        if constexpr (mode == SlopeMode::None) {
            return sham::VectorProperties<T>::get_zero();
        }

        if constexpr (mode == SlopeMode::VanLeer_f) {
            return slope_function_van_leer_f_form(sL, sR);
        }

        if constexpr (mode == SlopeMode::VanLeer_std) {
            return slope_function_van_leer_standard(sL, sR);
        }

        if constexpr (mode == SlopeMode::VanLeer_sym) {
            return slope_function_van_leer_symetric(sL, sR);
        }

        if constexpr (mode == SlopeMode::Minmod) {
            return slope_function_minmod(sL, sR);
        }
    }

    /**
     * @brief Get the 3d, slope limited gradient of a field
     *
     * @tparam T
     * @tparam Tvec
     * @tparam mode
     * @tparam ACCField
     * @param cell_global_id
     * @param delta_cell
     * @param graph_iter_xp
     * @param graph_iter_xm
     * @param graph_iter_yp
     * @param graph_iter_ym
     * @param graph_iter_zp
     * @param graph_iter_zm
     * @param field_access
     * @return std::array<T, 3>
     */
    template<class T, class Tvec, SlopeMode mode, class ACCField>
    inline std::array<T, 3> get_3d_grad(
        const u32 cell_global_id,
        const shambase::VecComponent<Tvec> delta_cell,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access) {

        auto get_avg_neigh = [&](auto &graph_links) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });
            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        };

        T W_i  = field_access(cell_global_id);
        T W_xp = get_avg_neigh(graph_iter_xp);
        T W_xm = get_avg_neigh(graph_iter_xm);
        T W_yp = get_avg_neigh(graph_iter_yp);
        T W_ym = get_avg_neigh(graph_iter_ym);
        T W_zp = get_avg_neigh(graph_iter_zp);
        T W_zm = get_avg_neigh(graph_iter_zm);

        T delta_W_x_p = W_xp - W_i;
        T delta_W_y_p = W_yp - W_i;
        T delta_W_z_p = W_zp - W_i;

        T delta_W_x_m = W_i - W_xm;
        T delta_W_y_m = W_i - W_ym;
        T delta_W_z_m = W_i - W_zm;

        T fact = 1. / T(delta_cell);

        T lim_slope_W_x = slope_function<T, mode>(delta_W_x_m * fact, delta_W_x_p * fact);
        T lim_slope_W_y = slope_function<T, mode>(delta_W_y_m * fact, delta_W_y_p * fact);
        T lim_slope_W_z = slope_function<T, mode>(delta_W_z_m * fact, delta_W_z_p * fact);

        return {lim_slope_W_x, lim_slope_W_y, lim_slope_W_z};
    }
} // namespace

template<class Tvec, class TgridVec>
template<SlopeMode mode>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::
    _compute_grad_rho_van_leer() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tvec> result
        = utility.make_compute_field<Tvec>("gradient rho", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");

    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

            sham::DeviceBuffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_ghost);

            AMRGraph &graph_neigh_xp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
            AMRGraph &graph_neigh_xm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
            AMRGraph &graph_neigh_yp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
            AMRGraph &graph_neigh_ym
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
            AMRGraph &graph_neigh_zp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
            AMRGraph &graph_neigh_zm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

            sham::DeviceBuffer<Tscal> &block_cell_sizes
                = shambase::get_check_ref(storage.block_cell_sizes)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();
            sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
                = shambase::get_check_ref(storage.cell0block_aabb_lower)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();

            sham::EventList depends_list;

            auto acc_aabb_block_lower = cell0block_aabb_lower.get_read_access(depends_list);
            auto acc_aabb_cell_size   = block_cell_sizes.get_read_access(depends_list);
            auto rho                  = buf_rho.get_read_access(depends_list);
            auto grad_rho             = result.get_buf(id).get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
                AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
                AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
                AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
                AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
                AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

                u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

                shambase::parralel_for(cgh, cell_count, "compute_grad_rho", [=](u64 gid) {
                    const u32 cell_global_id = (u32) gid;

                    const u32 block_id    = cell_global_id / AMRBlock::block_size;
                    const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                    Tscal delta_cell = acc_aabb_cell_size[block_id];

                    auto result = get_3d_grad<Tscal, Tvec, mode>(
                        cell_global_id,
                        delta_cell,
                        graph_iter_xp,
                        graph_iter_xm,
                        graph_iter_yp,
                        graph_iter_ym,
                        graph_iter_zp,
                        graph_iter_zm,
                        [=](u32 id) {
                            return rho[id];
                        });

                    grad_rho[cell_global_id] = {result[0], result[1], result[2]};
                });
            });

            cell0block_aabb_lower.complete_event_state(e);
            block_cell_sizes.complete_event_state(e);
            buf_rho.complete_event_state(e);
            result.get_buf(id).complete_event_state(e);
        });

    storage.grad_rho.set(std::move(result));
}

template<class Tvec, class TgridVec>
template<SlopeMode mode>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::_compute_grad_v_van_leer() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tvec> resultx
        = utility.make_compute_field<Tvec>("gradient dx v", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });
    shamrock::ComputeField<Tvec> resulty
        = utility.make_compute_field<Tvec>("gradient dy v", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });
    shamrock::ComputeField<Tvec> resultz
        = utility.make_compute_field<Tvec>("gradient dz v", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

            sham::DeviceBuffer<Tvec> &buf_vel = shambase::get_check_ref(storage.vel).get_buf(id);

            AMRGraph &graph_neigh_xp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
            AMRGraph &graph_neigh_xm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
            AMRGraph &graph_neigh_yp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
            AMRGraph &graph_neigh_ym
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
            AMRGraph &graph_neigh_zp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
            AMRGraph &graph_neigh_zm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

            sham::DeviceBuffer<Tscal> &block_cell_sizes
                = shambase::get_check_ref(storage.block_cell_sizes)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();
            sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
                = shambase::get_check_ref(storage.cell0block_aabb_lower)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();

            sham::EventList depends_list;

            auto acc_aabb_block_lower = cell0block_aabb_lower.get_read_access(depends_list);
            auto acc_aabb_cell_size   = block_cell_sizes.get_read_access(depends_list);
            auto vel                  = buf_vel.get_read_access(depends_list);
            auto dx_vel               = resultx.get_buf(id).get_write_access(depends_list);
            auto dy_vel               = resulty.get_buf(id).get_write_access(depends_list);
            auto dz_vel               = resultz.get_buf(id).get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
                AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
                AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
                AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
                AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
                AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

                u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

                Tscal dxfact = solver_config.grid_coord_to_pos_fact;

                shambase::parralel_for(cgh, cell_count, "compute_grad_v", [=](u64 gid) {
                    const u32 cell_global_id = (u32) gid;

                    const u32 block_id    = cell_global_id / AMRBlock::block_size;
                    const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                    Tscal delta_cell = acc_aabb_cell_size[block_id];

                    auto result = get_3d_grad<Tvec, Tvec, mode>(
                        cell_global_id,
                        delta_cell,
                        graph_iter_xp,
                        graph_iter_xm,
                        graph_iter_yp,
                        graph_iter_ym,
                        graph_iter_zp,
                        graph_iter_zm,
                        [=](u32 id) {
                            return vel[id];
                        });

                    dx_vel[cell_global_id] = result[0];
                    dy_vel[cell_global_id] = result[1];
                    dz_vel[cell_global_id] = result[2];
                });
            });

            cell0block_aabb_lower.complete_event_state(e);
            block_cell_sizes.complete_event_state(e);
            buf_vel.complete_event_state(e);
            resultx.get_buf(id).complete_event_state(e);
            resulty.get_buf(id).complete_event_state(e);
            resultz.get_buf(id).complete_event_state(e);
        });

    storage.dx_v.set(std::move(resultx));
    storage.dy_v.set(std::move(resulty));
    storage.dz_v.set(std::move(resultz));
}

template<class Tvec, class TgridVec>
template<SlopeMode mode>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::_compute_grad_P_van_leer() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tvec> result = utility.make_compute_field<Tvec>(
        "gradient rho rhoetot", AMRBlock::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        });

    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

            sham::DeviceBuffer<Tscal> &buf_press
                = shambase::get_check_ref(storage.press).get_buf(id);

            AMRGraph &graph_neigh_xp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
            AMRGraph &graph_neigh_xm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
            AMRGraph &graph_neigh_yp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
            AMRGraph &graph_neigh_ym
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
            AMRGraph &graph_neigh_zp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
            AMRGraph &graph_neigh_zm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

            sham::DeviceBuffer<Tscal> &block_cell_sizes
                = shambase::get_check_ref(storage.block_cell_sizes)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();
            sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
                = shambase::get_check_ref(storage.cell0block_aabb_lower)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();

            sham::EventList depends_list;

            auto acc_aabb_block_lower = cell0block_aabb_lower.get_read_access(depends_list);
            auto acc_aabb_cell_size   = block_cell_sizes.get_read_access(depends_list);
            auto press                = buf_press.get_read_access(depends_list);
            auto grad_P               = result.get_buf(id).get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
                AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
                AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
                AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
                AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
                AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

                u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

                Tscal dxfact = solver_config.grid_coord_to_pos_fact;
                Tscal gamma  = solver_config.eos_gamma;

                shambase::parralel_for(cgh, cell_count, "compute_grad_rho", [=](u64 gid) {
                    const u32 cell_global_id = (u32) gid;

                    const u32 block_id    = cell_global_id / AMRBlock::block_size;
                    const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                    Tscal delta_cell = acc_aabb_cell_size[block_id];

                    auto result = get_3d_grad<Tscal, Tvec, mode>(
                        cell_global_id,
                        delta_cell,
                        graph_iter_xp,
                        graph_iter_xm,
                        graph_iter_yp,
                        graph_iter_ym,
                        graph_iter_zp,
                        graph_iter_zm,
                        [=](u32 id) {
                            return press[id];
                        });

                    grad_P[cell_global_id] = {result[0], result[1], result[2]};
                });
            });

            cell0block_aabb_lower.complete_event_state(e);
            block_cell_sizes.complete_event_state(e);
            buf_press.complete_event_state(e);
            result.get_buf(id).complete_event_state(e);
        });

    storage.grad_P.set(std::move(result));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::
    compute_grad_rho_van_leer() {
    if (solver_config.slope_config == SlopeMode::None) {
        _compute_grad_rho_van_leer<SlopeMode::None>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_f) {
        _compute_grad_rho_van_leer<SlopeMode::VanLeer_f>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_std) {
        _compute_grad_rho_van_leer<SlopeMode::VanLeer_std>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_sym) {
        _compute_grad_rho_van_leer<SlopeMode::VanLeer_sym>();
    } else if (solver_config.slope_config == SlopeMode::Minmod) {
        _compute_grad_rho_van_leer<SlopeMode::Minmod>();
    } else {
        shambase::throw_unimplemented();
    }
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::compute_grad_v_van_leer() {
    if (solver_config.slope_config == SlopeMode::None) {
        _compute_grad_v_van_leer<SlopeMode::None>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_f) {
        _compute_grad_v_van_leer<SlopeMode::VanLeer_f>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_std) {
        _compute_grad_v_van_leer<SlopeMode::VanLeer_std>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_sym) {
        _compute_grad_v_van_leer<SlopeMode::VanLeer_sym>();
    } else if (solver_config.slope_config == SlopeMode::Minmod) {
        _compute_grad_v_van_leer<SlopeMode::Minmod>();
    } else {
        shambase::throw_unimplemented();
    }
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::compute_grad_P_van_leer() {
    if (solver_config.slope_config == SlopeMode::None) {
        _compute_grad_P_van_leer<SlopeMode::None>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_f) {
        _compute_grad_P_van_leer<SlopeMode::VanLeer_f>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_std) {
        _compute_grad_P_van_leer<SlopeMode::VanLeer_std>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_sym) {
        _compute_grad_P_van_leer<SlopeMode::VanLeer_sym>();
    } else if (solver_config.slope_config == SlopeMode::Minmod) {
        _compute_grad_P_van_leer<SlopeMode::Minmod>();
    } else {
        shambase::throw_unimplemented();
    }
}

template<class Tvec, class TgridVec>
template<SlopeMode mode>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::
    _compute_grad_rho_dust_van_leer() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;
    u32 ndust        = solver_config.dust_config.ndust;
    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tvec> result = utility.make_compute_field<Tvec>(
        "gradient rho dust", ndust * AMRBlock::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_dust_ghost                            = ghost_layout.get_field_idx<Tscal>("rho_dust");

    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

            sham::DeviceBuffer<Tscal> &buf_rho_dust
                = mpdat.pdat.get_field_buf_ref<Tscal>(irho_dust_ghost);

            AMRGraph &graph_neigh_xp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
            AMRGraph &graph_neigh_xm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
            AMRGraph &graph_neigh_yp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
            AMRGraph &graph_neigh_ym
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
            AMRGraph &graph_neigh_zp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
            AMRGraph &graph_neigh_zm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

            sham::DeviceBuffer<Tscal> &block_cell_sizes
                = shambase::get_check_ref(storage.block_cell_sizes)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();
            sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
                = shambase::get_check_ref(storage.cell0block_aabb_lower)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();

            sham::EventList depends_list;

            auto acc_aabb_block_lower = cell0block_aabb_lower.get_read_access(depends_list);
            auto acc_aabb_cell_size   = block_cell_sizes.get_read_access(depends_list);
            auto rho_dust             = buf_rho_dust.get_read_access(depends_list);
            auto grad_rho_dust        = result.get_buf(id).get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
                AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
                AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
                AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
                AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
                AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

                u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;
                u32 nvar_dust  = ndust;

                shambase::parralel_for(
                    cgh, cell_count * nvar_dust, "compute_grad_rho_dust", [=](u64 gid) {
                        const u32 tmp_gid        = (u32) gid;
                        const u32 cell_global_id = tmp_gid / nvar_dust;
                        const u32 ndust_off_loc  = tmp_gid % nvar_dust;

                        const u32 block_id    = cell_global_id / AMRBlock::block_size;
                        const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                        Tscal delta_cell = acc_aabb_cell_size[block_id];

                        auto result = get_3d_grad<Tscal, Tvec, mode>(
                            cell_global_id,
                            delta_cell,
                            graph_iter_xp,
                            graph_iter_xm,
                            graph_iter_yp,
                            graph_iter_ym,
                            graph_iter_zp,
                            graph_iter_zm,
                            [=](u32 id) {
                                return rho_dust[nvar_dust * id + ndust_off_loc];
                            });
                        grad_rho_dust[nvar_dust * cell_global_id + ndust_off_loc]
                            = {result[0], result[1], result[2]};
                    });
            });

            cell0block_aabb_lower.complete_event_state(e);
            block_cell_sizes.complete_event_state(e);
            buf_rho_dust.complete_event_state(e);
            result.get_buf(id).complete_event_state(e);
        });

    storage.grad_rho_dust.set(std::move(result));
}

template<class Tvec, class TgridVec>
template<SlopeMode mode>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::
    _compute_grad_v_dust_van_leer() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;
    u32 ndust        = solver_config.dust_config.ndust;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tvec> resultx = utility.make_compute_field<Tvec>(
        "gradient dx v dust", ndust * AMRBlock::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        });
    shamrock::ComputeField<Tvec> resulty = utility.make_compute_field<Tvec>(
        "gradient dy v dust", ndust * AMRBlock::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        });
    shamrock::ComputeField<Tvec> resultz = utility.make_compute_field<Tvec>(
        "gradient dz v dust", ndust * AMRBlock::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        });

    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

            sham::DeviceBuffer<Tvec> &buf_vel_dust
                = shambase::get_check_ref(storage.vel_dust).get_buf(id);

            AMRGraph &graph_neigh_xp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
            AMRGraph &graph_neigh_xm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
            AMRGraph &graph_neigh_yp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
            AMRGraph &graph_neigh_ym
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
            AMRGraph &graph_neigh_zp
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
            AMRGraph &graph_neigh_zm
                = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

            sham::DeviceBuffer<Tscal> &block_cell_sizes
                = shambase::get_check_ref(storage.block_cell_sizes)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();
            sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
                = shambase::get_check_ref(storage.cell0block_aabb_lower)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();

            sham::EventList depends_list;

            auto acc_aabb_block_lower = cell0block_aabb_lower.get_read_access(depends_list);
            auto acc_aabb_cell_size   = block_cell_sizes.get_read_access(depends_list);
            auto vel_dust             = buf_vel_dust.get_read_access(depends_list);
            auto dx_vel_dust          = resultx.get_buf(id).get_write_access(depends_list);
            auto dy_vel_dust          = resulty.get_buf(id).get_write_access(depends_list);
            auto dz_vel_dust          = resultz.get_buf(id).get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
                AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
                AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
                AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
                AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
                AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

                u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

                u32 nvar_dust = ndust;
                shambase::parralel_for(
                    cgh, cell_count * nvar_dust, "compute_grad_v_dust", [=](u64 gid) {
                        const u32 tmp_gid        = (u32) gid;
                        const u32 cell_global_id = tmp_gid / nvar_dust;
                        const u32 ndust_off_loc  = tmp_gid % nvar_dust;

                        const u32 block_id    = cell_global_id / AMRBlock::block_size;
                        const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                        Tscal delta_cell = acc_aabb_cell_size[block_id];

                        auto result = get_3d_grad<Tvec, Tvec, mode>(
                            cell_global_id,
                            delta_cell,
                            graph_iter_xp,
                            graph_iter_xm,
                            graph_iter_yp,
                            graph_iter_ym,
                            graph_iter_zp,
                            graph_iter_zm,
                            [=](u32 id) {
                                return vel_dust[id * nvar_dust + ndust_off_loc];
                            });

                        dx_vel_dust[cell_global_id * nvar_dust + ndust_off_loc] = result[0];
                        dy_vel_dust[cell_global_id * nvar_dust + ndust_off_loc] = result[1];
                        dz_vel_dust[cell_global_id * nvar_dust + ndust_off_loc] = result[2];
                    });
            });

            cell0block_aabb_lower.complete_event_state(e);
            block_cell_sizes.complete_event_state(e);
            buf_vel_dust.complete_event_state(e);
            resultx.get_buf(id).complete_event_state(e);
            resulty.get_buf(id).complete_event_state(e);
            resultz.get_buf(id).complete_event_state(e);
        });

    storage.dx_v_dust.set(std::move(resultx));
    storage.dy_v_dust.set(std::move(resulty));
    storage.dz_v_dust.set(std::move(resultz));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::
    compute_grad_rho_dust_van_leer() {
    if (solver_config.slope_config == SlopeMode::None) {
        _compute_grad_rho_dust_van_leer<SlopeMode::None>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_f) {
        _compute_grad_rho_dust_van_leer<SlopeMode::VanLeer_f>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_std) {
        _compute_grad_rho_dust_van_leer<SlopeMode::VanLeer_std>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_sym) {
        _compute_grad_rho_dust_van_leer<SlopeMode::VanLeer_sym>();
    } else if (solver_config.slope_config == SlopeMode::Minmod) {
        _compute_grad_rho_dust_van_leer<SlopeMode::Minmod>();
    } else {
        shambase::throw_unimplemented();
    }
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeGradient<Tvec, TgridVec>::
    compute_grad_v_dust_van_leer() {
    if (solver_config.slope_config == SlopeMode::None) {
        _compute_grad_v_dust_van_leer<SlopeMode::None>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_f) {
        _compute_grad_v_dust_van_leer<SlopeMode::VanLeer_f>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_std) {
        _compute_grad_v_dust_van_leer<SlopeMode::VanLeer_std>();
    } else if (solver_config.slope_config == SlopeMode::VanLeer_sym) {
        _compute_grad_v_dust_van_leer<SlopeMode::VanLeer_sym>();
    } else if (solver_config.slope_config == SlopeMode::Minmod) {
        _compute_grad_v_dust_van_leer<SlopeMode::Minmod>();
    } else {
        shambase::throw_unimplemented();
    }
}

template class shammodels::basegodunov::modules::ComputeGradient<f64_3, i64_3>;
