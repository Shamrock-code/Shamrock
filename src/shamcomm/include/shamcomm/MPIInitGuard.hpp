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
 * @file MPIInitGuard.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief RAII wrapper around MPI_Init and MPI_Finalize
 */

namespace shamcomm {

    /**
     * @brief RAII guard of the MPI library lifetime
     *
     * The constructor initializes MPI if it is not already initialized. The destructor (or
     * `close()`) finalizes MPI if and only if this guard started it.
     *
     * Copy and move are disabled: a process may initialize MPI only once, and ownership of that
     * session must stay unique.
     */
    class MPIInitGuard {
        public:
        /**
         * @brief Initialize MPI if it is not already initialized
         *
         * @param argc Argument count pointer forwarded to MPI_Init (may be nullptr)
         * @param argv Argument vector pointer forwarded to MPI_Init (may be nullptr)
         */
        explicit MPIInitGuard(int *argc = nullptr, char ***argv = nullptr);

        /// Finalize MPI if this guard owns the session
        ~MPIInitGuard();

        MPIInitGuard(const MPIInitGuard &)            = delete;
        MPIInitGuard &operator=(const MPIInitGuard &) = delete;
        MPIInitGuard(MPIInitGuard &&)                 = delete;
        MPIInitGuard &operator=(MPIInitGuard &&)      = delete;

        /**
         * @brief Finalize MPI early if this guard owns the session
         *
         * Idempotent: a second call (including the destructor after `close()`) is a no-op.
         */
        void close();

        /**
         * @brief Whether this guard currently owns an active MPI session
         *
         * @return true if this guard started MPI and has not closed it yet
         */
        bool is_active() const;

        private:
        bool owns_mpi = false;
    };

} // namespace shamcomm
