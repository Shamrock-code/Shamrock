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
 * @file GSPHGhostHandler.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief GSPH-specific ghost handler using Newtonian physics field names
 */

#include "shambase/DistributedData.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/solvergraph/ExchangeGhostField.hpp"
#include "shamrock/solvergraph/ExchangeGhostLayer.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamsys/NodeInstance.hpp"
#include <variant>

namespace shammodels::gsph {

    template<class vec>
    struct GSPHGhostHandlerConfig {

        using Tscal = shambase::VecComponent<vec>;

        struct Free {};
        struct Periodic {};
        struct ShearingPeriodic {
            i32_3 shear_base;
            i32_3 shear_dir;
            Tscal shear_value;
            Tscal shear_speed;
        };

        using Variant = std::variant<Free, Periodic, ShearingPeriodic>;
    };

    template<class vec>
    class GSPHGhostHandler {

        using CfgClass = GSPHGhostHandlerConfig<vec>;
        using Config   = typename CfgClass::Variant;

        PatchScheduler &sched;
        Config ghost_config;

        public:
        using flt                = shambase::VecComponent<vec>;
        static constexpr u32 dim = shambase::VectorProperties<vec>::dimension;
        using per_index          = sycl::vec<i32, dim>;

        struct InterfaceBuildInfos {
            vec offset;
            vec offset_speed;
            per_index periodicity_index;
            shammath::CoordRange<vec> cut_volume;
            flt volume_ratio;
        };

        struct InterfaceIdTable {
            InterfaceBuildInfos build_infos;
            sham::DeviceBuffer<u32> ids_interf;
            f64 part_cnt_ratio;
        };

        using GeneratorMap = shambase::DistributedDataShared<InterfaceBuildInfos>;

        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &xyzh_ghost_layout;

        std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> patch_rank_owner;

        GSPHGhostHandler(
            PatchScheduler &sched,
            Config ghost_config,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> patch_rank_owner,
            std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &xyzh_ghost_layout)
            : sched(sched), ghost_config(ghost_config), patch_rank_owner(patch_rank_owner),
              xyzh_ghost_layout(xyzh_ghost_layout) {}

        GeneratorMap find_interfaces(
            SerialPatchTree<vec> &sptree,
            shamrock::patch::PatchtreeField<flt> &int_range_max_tree,
            shamrock::patch::PatchField<flt> &int_range_max);

        shambase::DistributedDataShared<InterfaceIdTable> gen_id_table_interfaces(
            GeneratorMap &&gen);

        void gen_debug_patch_ghost(shambase::DistributedDataShared<InterfaceIdTable> &interf_info);

        using CacheMap = shambase::DistributedDataShared<InterfaceIdTable>;

        CacheMap make_interface_cache(
            SerialPatchTree<vec> &sptree,
            shamrock::patch::PatchtreeField<flt> &int_range_max_tree,
            shamrock::patch::PatchField<flt> &int_range_max) {
            StackEntry stack_loc{};

            return gen_id_table_interfaces(
                find_interfaces(sptree, int_range_max_tree, int_range_max));
        }

        template<class T>
        shambase::DistributedDataShared<T> build_interface_native(
            shambase::DistributedDataShared<InterfaceIdTable> &builder,
            std::function<T(u64, u64, InterfaceBuildInfos, sham::DeviceBuffer<u32> &, u32)> fct) {
            StackEntry stack_loc{};

            return builder.template map<T>([&](u64 sender,
                                               u64 receiver,
                                               InterfaceIdTable &build_table) {
                if (build_table.ids_interf.get_size() == 0) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "there is an empty id table in the interface, it should have been removed");
                }

                return fct(
                    sender,
                    receiver,
                    build_table.build_infos,
                    build_table.ids_interf,
                    build_table.ids_interf.get_size());
            });
        }

        template<class T>
        void modify_interface_native(
            shambase::DistributedDataShared<InterfaceIdTable> &builder,
            shambase::DistributedDataShared<T> &mod,
            std::function<void(u64, u64, InterfaceBuildInfos, sham::DeviceBuffer<u32> &, u32, T &)>
                fct) {
            StackEntry stack_loc{};

            struct Args {
                u64 sender;
                u64 receiver;
                InterfaceIdTable &build_table;
            };

            std::vector<Args> vecarg;

            builder.for_each([&](u64 sender, u64 receiver, InterfaceIdTable &build_table) {
                if (build_table.ids_interf.get_size() == 0) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "there is an empty id table in the interface, it should have been removed");
                }

                vecarg.push_back({sender, receiver, build_table});
            });

            u32 i = 0;
            mod.for_each([&](u64 sender, u64 receiver, T &ref) {
                InterfaceIdTable &build_table = vecarg[i].build_table;

                fct(sender,
                    receiver,
                    build_table.build_infos,
                    build_table.ids_interf,
                    build_table.ids_interf.get_size(),
                    ref);

                i++;
            });
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        // interface generation/communication utility //////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////

        inline shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>
        build_position_interf_field(shambase::DistributedDataShared<InterfaceIdTable> &builder) {
            StackEntry stack_loc{};

            const u32 ixyz   = sched.pdl().template get_field_idx<vec>("xyz");
            const u32 ihpart = sched.pdl().template get_field_idx<flt>("hpart");

            // Get field indices from xyzh_ghost_layout for accessing ghost data
            const u32 ixyz_ghost   = xyzh_ghost_layout->template get_field_idx<vec>("xyz");
            const u32 ihpart_ghost = xyzh_ghost_layout->template get_field_idx<flt>("hpart");

            return build_interface_native<shamrock::patch::PatchDataLayer>(
                builder,
                [&, ixyz_ghost, ihpart_ghost](
                    u64 sender,
                    u64 /*receiver*/,
                    InterfaceBuildInfos binfo,
                    sham::DeviceBuffer<u32> &buf_idx,
                    u32 cnt) {
                    using namespace shamrock::patch;

                    PatchDataLayer &sender_pdat = sched.patch_data.get_pdat(sender);

                    shamrock::patch::PatchDataLayer ret(xyzh_ghost_layout);

                    sender_pdat.get_field<vec>(ixyz).append_subset_to(
                        buf_idx, cnt, ret.get_field<vec>(ixyz_ghost));
                    sender_pdat.get_field<flt>(ihpart).append_subset_to(
                        buf_idx, cnt, ret.get_field<flt>(ihpart_ghost));

                    ret.get_field<vec>(ixyz_ghost).apply_offset(binfo.offset);

                    return ret;
                });
        }

        inline shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> communicate_pdat(
            const std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &pdl_ptr,
            shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> &&interf) {
            StackEntry stack_loc{};

            std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> exchange_gz_edge
                = std::make_shared<shamrock::solvergraph::PatchDataLayerDDShared>("", "");

            exchange_gz_edge->patchdatas
                = std::forward<shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>>(
                    interf);

            std::shared_ptr<shamrock::solvergraph::ExchangeGhostLayer> exchange_gz_node
                = std::make_shared<shamrock::solvergraph::ExchangeGhostLayer>(pdl_ptr);
            exchange_gz_node->set_edges(this->patch_rank_owner, exchange_gz_edge);

            exchange_gz_node->evaluate();

            shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> recv_dat;
            recv_dat = std::move(exchange_gz_edge->patchdatas);

            return recv_dat;
        }

        template<class T>
        inline shambase::DistributedDataShared<PatchDataField<T>> communicate_pdatfield(
            shambase::DistributedDataShared<PatchDataField<T>> &&interf, u32 nvar) {
            StackEntry stack_loc{};

            std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<T>> exchange_gz_edge
                = std::make_shared<shamrock::solvergraph::PatchDataFieldDDShared<T>>("", "");

            exchange_gz_edge->patchdata_fields
                = std::forward<shambase::DistributedDataShared<PatchDataField<T>>>(interf);

            std::shared_ptr<shamrock::solvergraph::ExchangeGhostField<T>> exchange_gz_node
                = std::make_shared<shamrock::solvergraph::ExchangeGhostField<T>>();
            exchange_gz_node->set_edges(this->patch_rank_owner, exchange_gz_edge);

            exchange_gz_node->evaluate();

            shambase::DistributedDataShared<PatchDataField<T>> recv_dat;
            recv_dat = std::move(exchange_gz_edge->patchdata_fields);

            return recv_dat;
        }

        inline shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>
        build_communicate_positions(shambase::DistributedDataShared<InterfaceIdTable> &builder) {
            auto pos_interf = build_position_interf_field(builder);
            return communicate_pdat(xyzh_ghost_layout, std::move(pos_interf));
        }

        template<class T, class Tmerged>
        inline shambase::DistributedData<Tmerged> merge_native(
            shambase::DistributedDataShared<T> &&interfs,
            std::function<
                Tmerged(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &pdat)> init,
            std::function<void(Tmerged &, T &)> appender) {

            StackEntry stack_loc{};

            shambase::DistributedData<Tmerged> merge_f;

            sched.for_each_patchdata_nonempty(
                [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                    Tmerged tmp_merge = init(p, pdat);

                    interfs.for_each([&](u64 sender, u64 receiver, T &interface) {
                        if (receiver == p.id_patch) {
                            appender(tmp_merge, interface);
                        }
                    });

                    merge_f.add_obj(p.id_patch, std::move(tmp_merge));
                });

            return merge_f;
        }

        inline shambase::DistributedData<shamrock::patch::PatchDataLayer> merge_position_buf(
            shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> &&positioninterfs) {
            StackEntry stack_loc{};

            const u32 ixyz   = sched.pdl().template get_field_idx<vec>("xyz");
            const u32 ihpart = sched.pdl().template get_field_idx<flt>("hpart");

            // Get field indices from xyzh_ghost_layout for accessing ghost data
            const u32 ixyz_ghost   = xyzh_ghost_layout->template get_field_idx<vec>("xyz");
            const u32 ihpart_ghost = xyzh_ghost_layout->template get_field_idx<flt>("hpart");

            return merge_native<shamrock::patch::PatchDataLayer, shamrock::patch::PatchDataLayer>(
                std::forward<shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>>(
                    positioninterfs),
                [=](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                    PatchDataField<vec> &pos   = pdat.get_field<vec>(ixyz);
                    PatchDataField<flt> &hpart = pdat.get_field<flt>(ihpart);

                    shamrock::patch::PatchDataLayer ret(xyzh_ghost_layout);

                    ret.get_field<vec>(ixyz_ghost).insert(pos);
                    ret.get_field<flt>(ihpart_ghost).insert(hpart);
                    ret.check_field_obj_cnt_match();

                    return ret;
                },
                [](shamrock::patch::PatchDataLayer &merged, shamrock::patch::PatchDataLayer &pint) {
                    merged.insert_elements(pint);

                    merged.check_field_obj_cnt_match();
                });
        }

        inline shambase::DistributedData<shamrock::patch::PatchDataLayer>
        build_comm_merge_positions(shambase::DistributedDataShared<InterfaceIdTable> &builder) {
            auto pos_interf = build_position_interf_field(builder);
            return merge_position_buf(communicate_pdat(xyzh_ghost_layout, std::move(pos_interf)));
        }
    };

} // namespace shammodels::gsph
