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
 * @file BasicSPHGhosts.hpp
 * @author Guo (guo.yansong@optimind.tech)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
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

namespace shammodels::sph {

    template<class vec>
    struct BasicSPHGhostHandlerConfig {

        using Tscal = shambase::VecComponent<vec>;

        struct Free {};
        struct Periodic {};
        struct ShearingPeriodic {
            i32_3 shear_base;
            i32_3 shear_dir;
            Tscal shear_value;
            Tscal shear_speed;
        };
        /**
         * @brief Wall boundary configuration for ghost handling
         *
         * For wall boundaries, ghost communication is handled like Free boundaries,
         * but wall particles (created manually) provide neighbor support at boundaries.
         */
        struct Wall {};

        using Variant = std::variant<Free, Periodic, ShearingPeriodic, Wall>;
    };

    template<class vec>
    class BasicSPHGhostHandler {

        using CfgClass = BasicSPHGhostHandlerConfig<vec>;
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

        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> xyzh_ghost_layout;

        std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> patch_rank_owner;

        BasicSPHGhostHandler(
            PatchScheduler &sched,
            Config ghost_config,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> patch_rank_owner)
            : sched(sched), ghost_config(ghost_config), patch_rank_owner(patch_rank_owner) {
            xyzh_ghost_layout = std::make_shared<shamrock::patch::PatchDataLayerLayout>();
            xyzh_ghost_layout->add_field<vec>("xyz", 1);
            xyzh_ghost_layout->add_field<flt>("hpart", 1);
        }

        /**
         * @brief Find interfaces and their metadata
         *
         * @param sptree the serial patch tree
         * @param int_range_max_tree the smoothing length maximas hierachy
         * @param int_range_max the smoothing length maximas hierachy
         * @return GeneratorMap the generator map containing the metadata to build interfaces
         */
        GeneratorMap find_interfaces(
            SerialPatchTree<vec> &sptree,
            shamrock::patch::PatchtreeField<flt> &int_range_max_tree,
            shamrock::patch::PatchField<flt> &int_range_max);

        /**
         * @brief precompute interfaces members and cache result in the return
         *
         * @param gen
         * @return shambase::DistributedDataShared<InterfaceIdTable>
         */
        shambase::DistributedDataShared<InterfaceIdTable> gen_id_table_interfaces(
            GeneratorMap &&gen);

        void gen_debug_patch_ghost(shambase::DistributedDataShared<InterfaceIdTable> &interf_info);

        using CacheMap = shambase::DistributedDataShared<InterfaceIdTable>;

        /**
         * @brief utility to generate both the metadata and index tables
         *
         * @param sptree
         * @param int_range_max_tree
         * @param int_range_max
         * @return shambase::DistributedDataShared<InterfaceIdTable>
         */
        CacheMap make_interface_cache(
            SerialPatchTree<vec> &sptree,
            shamrock::patch::PatchtreeField<flt> &int_range_max_tree,
            shamrock::patch::PatchField<flt> &int_range_max) {
            StackEntry stack_loc{};

            return gen_id_table_interfaces(
                find_interfaces(sptree, int_range_max_tree, int_range_max));
        }

        /**
         * @brief native handle to generate interfaces
         * generate interfaces of type T (template arg) based on the provided function
         * @code{.cpp}
         * auto split_lists = grid.gen_splitlists(
         *     [&](u64 id_patch, Patch cur_p, PatchData &pdat) -> sycl::buffer<u32> {
         *          generate the buffer saying which cells should split
         *     }
         * );
         * @endcode
         *
         * @tparam T
         * @param builder
         * @param fct
         * @return shambase::DistributedDataShared<T>
         */
        template<class T>
        shambase::DistributedDataShared<T> build_interface_native(
            shambase::DistributedDataShared<InterfaceIdTable> &builder,
            std::function<T(u64, u64, InterfaceBuildInfos, sham::DeviceBuffer<u32> &, u32)> fct) {
            StackEntry stack_loc{};

            // clang-format off
            return builder.template map<T>([&](u64 sender, u64 receiver, InterfaceIdTable &build_table) {
                if (build_table.ids_interf.get_size() == 0) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "their is an empty id table in the interface, it should have been removed");
                }

                return fct(
                    sender,
                    receiver,
                    build_table.build_infos,
                    build_table.ids_interf,
                    build_table.ids_interf.get_size());

            });
            // clang-format on
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

            // clang-format off
            builder.for_each([&](u64 sender, u64 receiver, InterfaceIdTable &build_table) {
                if (build_table.ids_interf.get_size() == 0) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "their is an empty id table in the interface, it should have been removed");
                }

                vecarg.push_back({sender,receiver,build_table});
            });
            // clang-format on

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

        /**
         * @brief native handle to generate interfaces
         * generate interfaces of type T (template arg) based on the provided function
         * @code{.cpp}
         * // Example usage
         * @endcode
         *
         * @tparam T
         * @param builder
         * @param fct
         * @return shambase::DistributedDataShared<T>
         */
        template<class T>
        shambase::DistributedDataShared<T> build_interface_native_stagged(
            shambase::DistributedDataShared<InterfaceIdTable> &builder,
            std::function<T(u64, u64, InterfaceBuildInfos, sycl::buffer<u32> &, u32)> gen_1,
            std::function<void(u64, u64, InterfaceBuildInfos, sycl::buffer<u32> &, u32, T &)>
                modif) {

            StackEntry stack_loc{};

            struct Args {
                u64 sender;
                u64 receiver;
                InterfaceIdTable &build_table;
            };

            std::vector<Args> vecarg;

            shambase::DistributedDataShared<T> ret = builder.template map<
                T>([&](u64 sender, u64 receiver, InterfaceIdTable &build_table) {
                if (!bool(build_table.ids_interf)) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "their is an empty id table in the interface, it should have been removed");
                }

                vecarg.push_back({sender, receiver, build_table});

                return gen_1(
                    sender,
                    receiver,
                    build_table.build_infos,
                    *build_table.ids_interf,
                    build_table.ids_interf->size());
            });

            u32 i = 0;
            ret.for_each([&](u64 sender, u64 receiver, T &ref) {
                InterfaceIdTable &build_table = vecarg[i].build_table;

                modif(
                    sender,
                    receiver,
                    build_table.build_infos,
                    *build_table.ids_interf,
                    build_table.ids_interf->size(),
                    ref);

                i++;
            });

            return ret;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        // interface generation/communication utility //////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////

        inline shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>
        build_position_interf_field(shambase::DistributedDataShared<InterfaceIdTable> &builder) {
            StackEntry stack_loc{};

            const u32 ihpart = sched.pdl().template get_field_idx<flt>("hpart");

            return build_interface_native<shamrock::patch::PatchDataLayer>(
                builder,
                [&](u64 sender,
                    u64 /*receiver*/,
                    InterfaceBuildInfos binfo,
                    sham::DeviceBuffer<u32> &buf_idx,
                    u32 cnt) {
                    using namespace shamrock::patch;

                    PatchDataLayer &sender_pdat = sched.patch_data.get_pdat(sender);

                    shamrock::patch::PatchDataLayer ret(xyzh_ghost_layout);

                    sender_pdat.get_field<vec>(0).append_subset_to(
                        buf_idx, cnt, ret.get_field<vec>(0));
                    sender_pdat.get_field<flt>(ihpart).append_subset_to(
                        buf_idx, cnt, ret.get_field<flt>(1));

                    ret.get_field<vec>(0).apply_offset(binfo.offset);

                    return ret;
                });
        }

        inline shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> communicate_pdat(
            const std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &pdl_ptr,
            shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> &&interf) {
            StackEntry stack_loc{};

            // ----------------------------------------------------------------------------------------
            // temporary wrapper to slowly migrate to the new solvergraph
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> exchange_gz_edge
                = std::make_shared<shamrock::solvergraph::PatchDataLayerDDShared>("", "");

            exchange_gz_edge->patchdatas
                = std::forward<shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>>(
                    interf);

            std::shared_ptr<shamrock::solvergraph::ExchangeGhostLayer> exchange_gz_node
                = std::make_shared<shamrock::solvergraph::ExchangeGhostLayer>(pdl_ptr);
            exchange_gz_node->set_edges(this->patch_rank_owner, exchange_gz_edge);

            exchange_gz_node->evaluate();

            // ----------------------------------------------------------------------------------------

            shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> recv_dat;

#if false
            shamalgs::collective::serialize_sparse_comm<shamrock::patch::PatchDataLayer>(
                shamsys::instance::get_compute_scheduler_ptr(),
                std::forward<shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>>(
                    interf),
                recv_dat,
                [&](u64 id) {
                    return sched.get_patch_rank_owner(id);
                },
                [](shamrock::patch::PatchDataLayer &pdat) {
                    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
                    ser.allocate(pdat.serialize_buf_byte_size());
                    pdat.serialize_buf(ser);
                    return ser.finalize();
                },
                [&](sham::DeviceBuffer<u8> &&buf) {
                    // exchange the buffer held by the distrib data and give it to the serializer
                    shamalgs::SerializeHelper ser(
                        shamsys::instance::get_compute_scheduler_ptr(),
                        std::forward<sham::DeviceBuffer<u8>>(buf));
                    return shamrock::patch::PatchDataLayer::deserialize_buf(ser, pdl_ptr);
                });
#else
            recv_dat = std::move(exchange_gz_edge->patchdatas);
#endif

            return recv_dat;
        }

        template<class T>
        inline shambase::DistributedDataShared<PatchDataField<T>> communicate_pdatfield(
            shambase::DistributedDataShared<PatchDataField<T>> &&interf, u32 nvar) {
            StackEntry stack_loc{};

            // ----------------------------------------------------------------------------------------
            // temporary wrapper to slowly migrate to the new solvergraph
            std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<T>> exchange_gz_edge
                = std::make_shared<shamrock::solvergraph::PatchDataFieldDDShared<T>>("", "");

            exchange_gz_edge->patchdata_fields
                = std::forward<shambase::DistributedDataShared<PatchDataField<T>>>(interf);

            std::shared_ptr<shamrock::solvergraph::ExchangeGhostField<T>> exchange_gz_node
                = std::make_shared<shamrock::solvergraph::ExchangeGhostField<T>>();
            exchange_gz_node->set_edges(this->patch_rank_owner, exchange_gz_edge);

            exchange_gz_node->evaluate();

            // ----------------------------------------------------------------------------------------

            shambase::DistributedDataShared<PatchDataField<T>> recv_dat;

#if false
            shamalgs::collective::serialize_sparse_comm<PatchDataField<T>>(
                shamsys::instance::get_compute_scheduler_ptr(),
                std::forward<shambase::DistributedDataShared<PatchDataField<T>>>(interf),
                recv_dat,
                [&](u64 id) {
                    return sched.get_patch_rank_owner(id);
                },
                [](PatchDataField<T> &pdat) {
                    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
                    ser.allocate(pdat.serialize_full_byte_size());
                    pdat.serialize_full(ser);
                    return ser.finalize();
                },
                [&](sham::DeviceBuffer<u8> &&buf) {
                    // exchange the buffer held by the distrib data and give it to the serializer
                    shamalgs::SerializeHelper ser(
                        shamsys::instance::get_compute_scheduler_ptr(),
                        std::forward<sham::DeviceBuffer<u8>>(buf));
                    return PatchDataField<T>::deserialize_full(ser);
                });
#else
            recv_dat = std::move(exchange_gz_edge->patchdata_fields);
#endif

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

            const u32 ihpart = sched.pdl().template get_field_idx<flt>("hpart");

            return merge_native<shamrock::patch::PatchDataLayer, shamrock::patch::PatchDataLayer>(
                std::forward<shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>>(
                    positioninterfs),
                [=](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                    PatchDataField<vec> &pos   = pdat.get_field<vec>(0);
                    PatchDataField<flt> &hpart = pdat.get_field<flt>(ihpart);

                    shamrock::patch::PatchDataLayer ret(xyzh_ghost_layout);

                    ret.get_field<vec>(0).insert(pos);
                    ret.get_field<flt>(1).insert(hpart);
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

} // namespace shammodels::sph
