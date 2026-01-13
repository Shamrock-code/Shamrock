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
 * @file linspace.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Creating an array of N values between two values
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Create an array of N values between two values
     * @tparam Tval value type in the sequence (same type as bounds)
     * @param Rmin lower bound of the sequence
     * @param Rmax upper bound of the sequence
     * @param N number of samples to generate
     * @return a DeviceBuffer containing the sequence
     *
     * TODO: move to GPU
     */
    template<typename Tval>
    sham::DeviceBuffer<Tval> linspace(Tval Rmin, Tval Rmax, u32 N) {
        sham::DeviceBuffer<Tval> bins(N, shamsys::instance::get_compute_scheduler_ptr());
        Tval step = (Rmax - Rmin) / (N - 1);
        for (int i = 0; i < N; ++i) {
            bins.set_val_at_idx(i, Rmin + i * step);
        }
        return bins;
    }

} // namespace shamalgs::primitives
