// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "numericTests.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/memory.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include <numeric>

TestStart(
    Unittest, "shamalgs/numeric/details/stream_compact_fallback", streamcompactalg_fallback, 1) {
    TestStreamCompact test(
        (TestStreamCompact::vFunctionCall) shamalgs::numeric::details::stream_compact_fallback);
    test.check();
}

TestStart(
    Unittest, "shamalgs/numeric/stream_compact_fallback(usm)", streamcompactalgusm_fallback, 1) {
    TestStreamCompactUSM test(
        (TestStreamCompactUSM::vFunctionCall) shamalgs::numeric::details::stream_compact_fallback);
    test.check();
}
