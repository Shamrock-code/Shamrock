// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DataInserterUtility.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/patch/PatchData.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include <mpi.h>

namespace shamrock {

    /**
     * @brief Class to insert data in the PatchScheduler
     *
     * This class is used to insert data in the PatchScheduler.
     * It provides a way to push data into the scheduler and
     * handle the insertion of the data into the correct patches.
     *
     * @warning
     * This class must be used by all MPI ranks.
     */
    class DataInserterUtility {
        PatchScheduler &sched; ///< Scheduler to bind onto

        public:
        /**
         * @brief Constructor
         *
         * @param sched The PatchScheduler to work on.
         */
        DataInserterUtility(PatchScheduler &sched) : sched(sched) {}

        /**
         * @brief Pushes data into the scheduler.
         *
         * @warning This function must be called by all MPI ranks
         *
         * @todo use directly main field at id=0 and deduce type
         * @todo implement case we more object than the threshold are present
         *
         * @param pdat_ins The PatchData object containing the data to be inserted.
         * @param main_field_name The name of the main field.
         * @param split_threshold The threshold at which the data will be split.
         * @param load_balance_update A function to call after the insertion of the data.
         *                             This function should call the load balance algorithm.
         * @return The number of objects inserted.
         */
        template<class Tvec>
        u64 push_patch_data(
            shamrock::patch::PatchData &pdat_ins,
            std::string main_field_name,
            u32 split_threshold,
            std::function<void(void)> load_balance_update) {
            using namespace shamrock::patch;

            u64 pdat_ob_cnt = pdat_ins.get_obj_cnt();

            u64 sum_push = shamalgs::collective::allreduce_sum(pdat_ob_cnt);
            if (shamcomm::world_rank() == 0) {
                logger::info_ln("DataInserterUtility", "pushing data in scheduler, N =", sum_push);
            }

            if (pdat_ob_cnt < split_threshold) {
                bool should_insert = true;
                sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
                    if (should_insert) {
                        pdat.insert_elements(pdat_ins);
                        should_insert = false; // We insert only in first patch (no duplicates)
                    }
                });
            } else {
                shambase::throw_unimplemented("Not implemented yet please keep the obj count to be "
                                              "inserted below the split_threshold, sorrrrrry ...");
            }

            if (shamcomm::world_rank() == 0) {
                logger::info_ln("DataInserterUtility", "reattributing data ...");
            }

            shambase::Timer treatrib;
            treatrib.start();
            // move data into the corect patches
            SerialPatchTree<Tvec> sptree = SerialPatchTree<Tvec>::build(sched);
            ReattributeDataUtility reatrib(sched);
            sptree.attach_buf();
            reatrib.reatribute_patch_objects(sptree, main_field_name);
            sched.check_patchdata_locality_corectness();

            treatrib.end();
            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "DataInserterUtility", "reattributing data done in ", treatrib.get_time_str());
            }
            shamcomm::mpi::Barrier(MPI_COMM_WORLD);

            if (shamcomm::world_rank() == 0) {
                logger::info_ln("DataInserterUtility", "Compute load ...");
            }

            load_balance_update();

            shamcomm::mpi::Barrier(MPI_COMM_WORLD);

            if (shamcomm::world_rank() == 0) {
                logger::info_ln("DataInserterUtility", "run scheduler step ...");
            }

            sched.scheduler_step(false, false);
            sched.scheduler_step(true, true);

            return sum_push;
        }
    };

} // namespace shamrock
