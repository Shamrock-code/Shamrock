// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ParticleInjector.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/narrowing.hpp"
#include "shambase/numeric_limits.hpp"
#include "shambase/stacktrace.hpp"
#include "shammodels/common/injection/ParticleInjector.hpp"
#include "sham/format/format.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/CoordRange.hpp"
#include "shamrock/patch/PatchCoordTransform.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>

namespace shammodels::common::injection {

    template<class Tvec>
    ParticleInjector<Tvec>::ParticleInjector(
        const ParticleInjectionConfig<Tvec> &config, Tscal part_mass, Tscal hfact) {

        config.check_config();

        for (const auto &injector : config.injector_list) {
            if (const FallbackStream<Tvec> *s = std::get_if<FallbackStream<Tvec>>(&injector)) {
                streams.push_back(*s);
                schedules.emplace_back(*s, part_mass, hfact);
            } else {
                shambase::throw_unimplemented();
            }
        }
    }

    template<class Tvec>
    InjectionReport ParticleInjector<Tvec>::inject(
        PatchScheduler &sched, Tscal t_old, Tscal t_new) {

        __shamrock_stack_entry();

        if (!std::isfinite(t_new) || !(t_new >= t_old)) {
            shambase::throw_with_loc<std::runtime_error>(sham::format(
                "particle injection needs a finite timestep, got t_old = {} and t_new = {}",
                t_old,
                t_new));
        }

        if (!sched.pdl_old().has_field_name("part_id")) {
            shambase::throw_with_loc<std::runtime_error>(
                "particle injection tags the injected particles in the part_id field, which is "
                "missing from the patch data layout");
        }

        if (is_first_inject) {
            // the walls born before the first injection step are already in the simulation,
            // either from the run that wrote the dump we restart from, or because the injection
            // started before the simulation
            u64 skipped_walls = 0;
            for (auto &schedule : schedules) {
                skipped_walls += schedule.pop_born_walls(t_old).size();
            }
            if (skipped_walls > 0 && shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "ParticleInjection",
                    sham::format(
                        "assuming that the {} walls born at or before t = {} are already in the "
                        "simulation",
                        skipped_walls,
                        t_old));
            }
            is_first_inject = false;
        }

        remove_boundary_parts(sched);

        HostBatch batch;
        InjectionReport report;

        for (u32 k = 0; k < streams.size(); k++) {
            const FallbackStream<Tvec> &stream     = streams[k];
            FallbackStreamSchedule<Tvec> &schedule = schedules[k];

            u64 injector_bits = u64(k) << part_id_injector_shift;

            for (const StreamWall<Tscal> &wall : schedule.pop_born_walls(t_new)) {
                append_wall(
                    batch, k, wall, t_new, part_id_injected_flag | injector_bits, wall.first_part);
                report.born_walls++;
                report.born_parts += wall.part_count;
            }

            // the next walls wait upstream of the injection point as boundary particles
            if (t_new >= stream.rate.t_start()) {
                u64 slot = 0;
                for (const StreamWall<Tscal> &wall :
                     schedule.peek_unborn_walls(stream.n_boundary_walls)) {
                    append_wall(batch, k, wall, t_new, part_id_boundary_flag | injector_bits, slot);
                    slot += wall.part_count;
                }
                report.boundary_parts += slot;
            }
        }

        insert_batch(sched, batch);

        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "ParticleInjection",
                sham::format(
                    "t = {}, born walls = {} ({} particles), boundary particles = {}",
                    t_new,
                    report.born_walls,
                    report.born_parts,
                    report.boundary_parts));
        }

        return report;
    }

    template<class Tvec>
    auto ParticleInjector<Tvec>::get_max_dt(Tscal t) const -> Tscal {
        Tscal dt_max = shambase::get_infty<Tscal>();
        for (const FallbackStream<Tvec> &stream : streams) {
            Tscal t_start = stream.rate.t_start();
            if (t < t_start) {
                dt_max = sham::min(dt_max, t_start - t);
            }
        }
        return dt_max;
    }

    template<class Tvec>
    void ParticleInjector<Tvec>::append_wall(
        HostBatch &batch,
        u32 injector_idx,
        const StreamWall<Tscal> &wall,
        Tscal t,
        u64 id_base,
        u64 first_index) {

        const FallbackStream<Tvec> &stream     = streams[injector_idx];
        FallbackStreamSchedule<Tvec> &schedule = schedules[injector_idx];

        const auto &sites    = schedule.get_lattice(wall.lattice_n);
        const auto &[e1, e2] = schedule.get_cross_section_basis();

        // the wall moves at the injection velocity since it crossed the injection point
        Tvec center = stream.inject_pos + stream.inject_vel * (t - wall.t_birth);

        for (u32 i = 0; i < sites.size(); i++) {
            batch.xyz.push_back(center + sites[i][0] * e1 + sites[i][1] * e2);
            batch.vxyz.push_back(stream.inject_vel);
            batch.hpart.push_back(wall.hpart);
            batch.uint.push_back(stream.u_inject);
            batch.part_id.push_back(id_base | (first_index + i));
        }
    }

    template<class Tvec>
    void ParticleInjector<Tvec>::remove_boundary_parts(PatchScheduler &sched) {

        __shamrock_stack_entry();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = shambase::get_check_ref(dev_sched).get_queue();

        const u32 ipart_id = sched.pdl_old().get_field_idx<u64>("part_id");

        sham::DeviceBuffer<u32> keep_flag(0, dev_sched);
        sham::DeviceBuffer<u32> has_boundary(1, dev_sched);

        sched.for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                u32 obj_cnt = pdat.get_obj_cnt();

                sham::DeviceBuffer<u64> &buf_part_id = pdat.get_field_buf_ref<u64>(ipart_id);

                keep_flag.resize(obj_cnt);
                has_boundary.fill(0);

                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_part_id},
                    sham::MultiRef{keep_flag, has_boundary},
                    obj_cnt,
                    [boundary_flag = part_id_boundary_flag](
                        u32 id_a,
                        const u64 *__restrict part_id,
                        u32 *__restrict keep_flag,
                        u32 *__restrict has_boundary) {
                        bool is_boundary = (part_id[id_a] & boundary_flag) != 0;
                        keep_flag[id_a]  = is_boundary ? 0 : 1;

                        if (is_boundary) {
                            sycl::atomic_ref<
                                u32,
                                sycl::memory_order_relaxed,
                                sycl::memory_scope_device,
                                sycl::access::address_space::global_space>
                                atomic_has_boundary(has_boundary[0]);
                            atomic_has_boundary.fetch_or(1);
                        }
                    });

                if (has_boundary.get_val_at_idx(0) != 0) {
                    sham::DeviceBuffer<u32> keep_ids
                        = shamalgs::stream_compact(dev_sched, keep_flag, obj_cnt);
                    pdat.keep_ids(keep_ids, shambase::narrow_or_throw<u32>(keep_ids.get_size()));
                }
            });
    }

    template<class Tvec>
    void ParticleInjector<Tvec>::insert_batch(PatchScheduler &sched, HostBatch &batch) {

        __shamrock_stack_entry();

        using namespace shamrock::patch;

        if (batch.xyz.empty()) {
            return;
        }

        u32 part_count = shambase::narrow_or_throw<u32>(batch.xyz.size());

        // bounding box of the batch, to skip the patches that cannot contain any of its particles
        Tvec bmin = batch.xyz[0];
        Tvec bmax = batch.xyz[0];
        for (const Tvec &r : batch.xyz) {
            bmin = sham::min(bmin, r);
            bmax = sham::max(bmax, r);
        }

        // owner patch of each particle, searched among all the patches (not only the local ones)
        // so that every rank agrees on the particles that cannot be placed
        constexpr u64 no_owner = std::numeric_limits<u64>::max();
        std::vector<u64> owner(part_count, no_owner);

        PatchCoordTransform<Tvec> ptransf = sched.get_patch_transform<Tvec>();

        for (const Patch &p : sched.patch_list.global) {
            if (p.is_err_mode()) {
                continue;
            }

            shammath::CoordRange<Tvec> range = ptransf.to_obj_coord(p);
            if (!(sham::vec_compare_leq(range.lower, bmax)
                  && sham::vec_compare_leq(bmin, range.upper))) {
                continue;
            }

            for (u32 i = 0; i < part_count; i++) {
                if (owner[i] == no_owner && range.contain_pos(batch.xyz[i])) {
                    owner[i] = p.id_patch;
                }
            }
        }

        std::map<u64, std::vector<u32>> parts_per_patch;
        for (u32 i = 0; i < part_count; i++) {
            if (owner[i] == no_owner) {
                auto [box_min, box_max] = sched.get_box_volume<Tvec>();
                shambase::throw_with_loc<std::runtime_error>(sham::format(
                    "an injected particle at {} is outside of the simulation box [{}, {}), make "
                    "the box contain the injection region",
                    batch.xyz[i],
                    box_min,
                    box_max));
            }
            parts_per_patch[owner[i]].push_back(i);
        }

        // the batch is only copied to the device if this rank owns some of its particles
        std::optional<PatchDataLayer> batch_pdat;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sched.for_each_local_patchdata([&](const Patch &p, PatchDataLayer &pdat) {
            auto it = parts_per_patch.find(p.id_patch);
            if (it == parts_per_patch.end()) {
                return;
            }

            if (!batch_pdat) {
                PatchDataLayer &b = batch_pdat.emplace(sched.get_layout_ptr_old());
                b.resize(part_count);
                b.fields_raz();
                b.override_patch_field("xyz", batch.xyz);
                b.override_patch_field("vxyz", batch.vxyz);
                b.override_patch_field("hpart", batch.hpart);
                b.override_patch_field("part_id", batch.part_id);
                if (b.pdl().has_field_name("uint")) {
                    b.override_patch_field("uint", batch.uint);
                }
            }

            const std::vector<u32> &idxs = it->second;
            sham::DeviceBuffer<u32> idxs_buf(idxs.size(), dev_sched);
            idxs_buf.copy_from_stdvec(idxs);

            batch_pdat->append_subset_to(
                idxs_buf, shambase::narrow_or_throw<u32>(idxs.size()), pdat);
        });
    }

    template class ParticleInjector<f64_3>;

} // namespace shammodels::common::injection
