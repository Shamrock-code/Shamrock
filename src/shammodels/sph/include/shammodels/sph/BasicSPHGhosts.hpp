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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
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

        using Variant = std::variant<Free, Periodic, ShearingPeriodic>;
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

        BasicSPHGhostHandler(PatchScheduler &sched, Config ghost_config)
            : sched(sched), ghost_config(ghost_config) {}

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
        shambase::DistributedDataShared<InterfaceIdTable>
        gen_id_table_interfaces(GeneratorMap &&gen);

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
         * ~~~~~{.cpp}
         *
         * auto split_lists = grid.gen_splitlists(
         *     [&](u64 id_patch, Patch cur_p, PatchData &pdat) -> sycl::buffer<u32> {
         *          generate the buffer saying which cells should split
         *     }
         * );
         *
         * ~~~~~
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
         * ~~~~~{.cpp}
         *
         *
         *
         * ~~~~~
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

        struct PositionInterface {
            PatchDataField<vec> position_field;
            PatchDataField<flt> hpart_field;
            vec bmin;
            vec bmax;
        };

        inline shambase::DistributedDataShared<PositionInterface>
        build_position_interf_field(shambase::DistributedDataShared<InterfaceIdTable> &builder) {
            StackEntry stack_loc{};

            const u32 ihpart = sched.pdl().template get_field_idx<flt>("hpart");

            return build_interface_native<PositionInterface>(
                builder,
                [&](u64 sender,
                    u64 /*receiver*/,
                    InterfaceBuildInfos binfo,
                    sham::DeviceBuffer<u32> &buf_idx,
                    u32 cnt) {
                    using namespace shamrock::patch;
                    // clang-format off
                    PatchDataLayer & sender_pdat = sched
                        .patch_data
                        .get_pdat(sender);

                    PatchDataField<vec> pfield = sender_pdat
                        .get_field<vec>(0)
                        .make_new_from_subset(buf_idx, cnt);

                    PatchDataField<flt> hpartfield = sender_pdat
                        .get_field<flt>(ihpart)
                        .make_new_from_subset(buf_idx, cnt);
                    // clang-format on

                    pfield.apply_offset(binfo.offset);

                    vec bmin = binfo.cut_volume.lower + binfo.offset;
                    vec bmax = binfo.cut_volume.upper + binfo.offset;

                    return PositionInterface{std::move(pfield), std::move(hpartfield), bmin, bmax};
                });
        }

        inline shambase::DistributedDataShared<PositionInterface>
        communicate_positions(shambase::DistributedDataShared<PositionInterface> &&interf) {
            StackEntry stack_loc{};

            shambase::DistributedDataShared<PositionInterface> recv_dat;

            shamalgs::collective::serialize_sparse_comm<PositionInterface>(
                shamsys::instance::get_compute_scheduler_ptr(),
                std::forward<shambase::DistributedDataShared<PositionInterface>>(interf),
                recv_dat,
                [&](u64 id) {
                    return sched.get_patch_rank_owner(id);
                },
                [](PositionInterface &pos_interf) {
                    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());

                    shamalgs::SerializeSize size
                        = pos_interf.position_field.serialize_buf_byte_size();
                    size += pos_interf.hpart_field.serialize_buf_byte_size();
                    size += shamalgs::SerializeHelper::serialize_byte_size<vec>() * 2;

                    ser.allocate(size);

                    pos_interf.position_field.serialize_buf(ser);
                    pos_interf.hpart_field.serialize_buf(ser);
                    ser.write(pos_interf.bmin);
                    ser.write(pos_interf.bmax);

                    return ser.finalize();
                },
                [&](sham::DeviceBuffer<u8> &&buf) {
                    // exchange the buffer held by the distrib data and give it to the
                    // serializer
                    shamalgs::SerializeHelper ser(
                        shamsys::instance::get_compute_scheduler_ptr(),
                        std::forward<sham::DeviceBuffer<u8>>(buf));

                    PatchDataField<vec> f = PatchDataField<vec>::deserialize_buf(ser, "xyz", 1);
                    PatchDataField<flt> hpart
                        = PatchDataField<flt>::deserialize_buf(ser, "hpart", 1);

                    vec bmin, bmax;
                    ser.load(bmin);
                    ser.load(bmax);

                    return PositionInterface{std::move(f), std::move(hpart), bmin, bmax};
                });

            return recv_dat;
        }

        inline shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> communicate_pdat(
            const std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &pdl_ptr,
            shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> &&interf) {
            StackEntry stack_loc{};

            shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> recv_dat;

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

            return recv_dat;
        }

        template<class T>
        inline shambase::DistributedDataShared<PatchDataField<T>> communicate_pdatfield(
            shambase::DistributedDataShared<PatchDataField<T>> &&interf, u32 nvar) {
            StackEntry stack_loc{};

            shambase::DistributedDataShared<PatchDataField<T>> recv_dat;

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

            return recv_dat;
        }

        inline shambase::DistributedDataShared<PositionInterface>
        build_communicate_positions(shambase::DistributedDataShared<InterfaceIdTable> &builder) {
            auto pos_interf = build_position_interf_field(builder);
            return communicate_positions(std::move(pos_interf));
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

        struct PreStepMergedField {
            shammath::CoordRange<vec> bounds;
            u32 original_elements;
            u32 total_elements;
            PatchDataField<vec> field_pos;
            PatchDataField<flt> field_hpart;
        };

        inline shambase::DistributedData<PreStepMergedField>
        merge_position_buf(shambase::DistributedDataShared<PositionInterface> &&positioninterfs) {
            StackEntry stack_loc{};

            const u32 ihpart = sched.pdl().template get_field_idx<flt>("hpart");

            return merge_native<PositionInterface, PreStepMergedField>(
                std::forward<shambase::DistributedDataShared<PositionInterface>>(positioninterfs),
                [=](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                    PatchDataField<vec> &pos      = pdat.get_field<vec>(0);
                    PatchDataField<flt> &hpart    = pdat.get_field<flt>(ihpart);
                    vec bmax                      = pos.compute_max();
                    vec bmin                      = pos.compute_min();
                    u32 or_elem                   = pos.get_obj_cnt();
                    PatchDataField<vec> new_pos   = pos.duplicate();
                    PatchDataField<flt> new_hpart = hpart.duplicate();

                    u32 total_elements = or_elem;

                    return PreStepMergedField{
                        shammath::CoordRange<vec>{bmin, bmax},
                        or_elem,
                        total_elements,
                        std::move(new_pos),
                        std::move(new_hpart)};
                },
                [](PreStepMergedField &merged, PositionInterface &pint) {
                    merged.total_elements += pint.position_field.get_obj_cnt();
                    merged.field_pos.insert(pint.position_field);
                    merged.field_hpart.insert(pint.hpart_field);
                    merged.bounds.upper = sham::max(merged.bounds.upper, pint.bmax);
                    merged.bounds.lower = sham::min(merged.bounds.lower, pint.bmin);
                });
        }

        inline shambase::DistributedData<PreStepMergedField>
        build_comm_merge_positions(shambase::DistributedDataShared<InterfaceIdTable> &builder) {
            auto pos_interf = build_position_interf_field(builder);
            return merge_position_buf(communicate_positions(std::move(pos_interf)));
        }
    };

} // namespace shammodels::sph
