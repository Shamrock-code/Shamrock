// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CommRequests.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "CommProtocolException.hpp"
#include "ProtocolEnum.hpp"
#include "shamsys/MpiWrapper.hpp"
#include <vector>

namespace shamsys::comm {

    /**
     * @brief Class to handle a collection of MPI_Request(s)
     *
     */
    class CommRequests {

        std::vector<MPI_Request> rqs;

        public:
        /**
         * @brief add a request to the collection
         *
         * @param rq the mpi request
         */
        inline void push(MPI_Request rq) { rqs.push_back(rq); }

        /**
         * @brief call a mpi_wait for the request held by the list
         *
         */
        inline void wait_all() {

            std::vector<MPI_Status> st_lst(rqs.size());
            mpi::waitall(rqs.size(), rqs.data(), st_lst.data());

            rqs.clear();

            // TODO shall we return the status list in a CommStatus class ?
        }
    };

} // namespace shamsys::comm
