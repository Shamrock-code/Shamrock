// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Solver.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "disc_analysis.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/reduction.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamcomm/collectives.hpp"
#include "shammath/sphkernels.hpp"
#include <memory>
#include <stdexcept>
#include "shammodels/sph/Model.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <utility>
#include <vector>

template<class Tvec, template<class> class SPHKernel>
using analysis = shamanalysis::disc_analysis<Tvec, SPHKernel>;

template<class Tvec, template<class> class SPHKernel>
void analysis<Tvec, SPHKernel>::disc(PhantomDump &phdump) {
    StackEntry stack_loc{};

    bool has_coord_in_header = true;

    Tscal xmin, xmax, ymin, ymax, zmin, zmax;
    has_coord_in_header = phdump.has_header_entry("xmin");

    std::string log = "";

    std::vector<Tvec> xyz, vxyz;
    std::vector<Tscal> h, u, alpha;

    {
        std::vector<Tscal> x, y, z, vx, vy, vz;

        phdump.blocks[0].fill_vec("x", x);
        phdump.blocks[0].fill_vec("y", y);
        phdump.blocks[0].fill_vec("z", z);

        if (has_coord_in_header) {
            xmin = phdump.read_header_float<f64>("xmin");
            xmax = phdump.read_header_float<f64>("xmax");
            ymin = phdump.read_header_float<f64>("ymin");
            ymax = phdump.read_header_float<f64>("ymax");
            zmin = phdump.read_header_float<f64>("zmin");
            zmax = phdump.read_header_float<f64>("zmax");

            //resize_simulation_box({{xmin, ymin, zmin}, {xmax, ymax, zmax}});
        } else {
            Tscal box_tolerance = 1.2;

            xmin = *std::min_element(x.begin(), x.end());
            xmax = *std::max_element(x.begin(), x.end());
            ymin = *std::min_element(y.begin(), y.end());
            ymax = *std::max_element(y.begin(), y.end());
            zmin = *std::min_element(z.begin(), z.end());
            zmax = *std::max_element(z.begin(), z.end());

            Tvec bm = {xmin, ymin, zmin};
            Tvec bM = {xmax, ymax, zmax};

            Tvec center = (bm + bM) * 0.5;

            Tvec d = (bM - bm) * 0.5;

            // expand the box
            d *= box_tolerance;

            //resize_simulation_box({center - d, center + d});
        }

        phdump.blocks[0].fill_vec("h", h);

        phdump.blocks[0].fill_vec("vx", vx);
        phdump.blocks[0].fill_vec("vy", vy);
        phdump.blocks[0].fill_vec("vz", vz);

        phdump.blocks[0].fill_vec("u", u);
        phdump.blocks[0].fill_vec("alpha", alpha);

        for (u32 i = 0; i < x.size(); i++) {
            xyz.push_back({x[i], y[i], z[i]});
        }
        for (u32 i = 0; i < vx.size(); i++) {
            vxyz.push_back({vx[i], vy[i], vz[i]});
        }
    }

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    u32 sz_buf = sched.crit_patch_split * 4;

    u32 Ntot = xyz.size();

    std::vector<u64> insert_ranges;
    insert_ranges.push_back(0);
    for (u64 i = sz_buf; i < Ntot; i += sz_buf) {
        insert_ranges.push_back(i);
    }
    insert_ranges.push_back(Ntot);

    for (u64 krange = 0; krange < insert_ranges.size() - 1; krange++) {
        u64 start_id = insert_ranges[krange];
        u64 end_id   = insert_ranges[krange + 1];

        u64 Nloc = end_id - start_id;

        sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
            PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

            shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

            std::vector<u64> sel_index;
            for (u64 i = start_id; i < end_id; i++) {
                Tvec r   = xyz[i];
                Tscal h_ = h[i];
                if (patch_coord.contain_pos(r) && (h_ >= 0)) {
                    sel_index.push_back(i);
                }
            }

            if (sel_index.size() == 0) {
                return;
            }

            log += shambase::format(
                "\n  rank = {}  patch id={}, add N={} particles, coords = {} {}",
                shamcomm::world_rank(),
                p.id_patch,
                sel_index.size(),
                patch_coord.lower,
                patch_coord.upper);

            std::vector<Tvec> ins_xyz, ins_vxyz;
            std::vector<Tscal> ins_h, ins_u, ins_alpha;
            for (u64 i : sel_index) {
                ins_xyz.push_back(xyz[i]);
            }
            for (u64 i : sel_index) {
                ins_vxyz.push_back(vxyz[i]);
            }
            for (u64 i : sel_index) {
                ins_h.push_back(h[i]);
            }
            if (u.size() > 0) {
                for (u64 i : sel_index) {
                    ins_u.push_back(u[i]);
                }
            }
            if (alpha.size() > 0) {
                for (u64 i : sel_index) {
                    ins_alpha.push_back(alpha[i]);
                }
            }

            PatchData ptmp(sched.pdl);
            ptmp.resize(sel_index.size());
            ptmp.fields_raz();

            ptmp.override_patch_field("xyz", ins_xyz);
            ptmp.override_patch_field("vxyz", ins_vxyz);
            ptmp.override_patch_field("hpart", ins_h);

            if (ins_alpha.size() > 0) {
                ptmp.override_patch_field("alpha_AV", ins_alpha);
            }

            if (ins_u.size() > 0) {
                ptmp.override_patch_field("uint", ins_u);
            }

            pdat.insert_elements(ptmp);
        });

        sched.check_patchdata_locality_corectness();

        std::string log_gathered = "";
        shamcomm::gather_str(log, log_gathered);

        if (shamcomm::world_rank() == 0) {
            logger::info_ln("Model", "Push particles : ", log_gathered);
        }
        log = "";

        modules::ComputeLoadBalanceValue<Tvec, SPHKernel> (ctx, solver.solver_config, solver.storage).update_load_balancing();

        post_insert_data<Tvec>(sched);

        // add sinks

        PhantomDumpBlock &sink_block = phdump.blocks[1];
        {
            std::vector<Tscal> xsink, ysink, zsink;
            std::vector<Tscal> vxsink, vysink, vzsink;
            std::vector<Tscal> mass;
            std::vector<Tscal> Racc;
init_from_phantom_dumpmodels::sph::Model
            sink_block.fill_vec("x", xsink);
            sink_block.fill_vec("y", ysink);
            sink_block.fill_vec("z", zsink);
            sink_block.fill_vec("vx", vxsink);
            sink_block.fill_vec("vy", vysink);
            sink_block.fill_vec("vz", vzsink);
            sink_block.fill_vec("m", mass);
            sink_block.fill_vec("h", Racc);

            for (u32 i = 0; i < xsink.size(); i++) {
                add_sink(
                    mass[i],
                    {xsink[i], ysink[i], zsink[i]},
                    {vxsink[i], vysink[i], vzsink[i]},
                    Racc[i]);
            }
        }
    }
}