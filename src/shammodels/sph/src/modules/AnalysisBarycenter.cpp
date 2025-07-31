// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AnalysisBarycenter.cpp
 * @author David Fang (fang.david03@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/AnalysisBarycenter.hpp"

using namespace shammath;

template class shammodels::sph::modules::AnalysisBarycenter<f64_3, M4>;
template class shammodels::sph::modules::AnalysisBarycenter<f64_3, M6>;
template class shammodels::sph::modules::AnalysisBarycenter<f64_3, M8>;
