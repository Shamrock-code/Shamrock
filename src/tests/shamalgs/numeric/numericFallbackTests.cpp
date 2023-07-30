// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamalgs/memory.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include <numeric>


#include "numericTests.hpp"




TestStart(Unittest, "shamalgs/numeric/details/stream_compact_fallback", streamcompactalg_fallback, 1){
    TestStreamCompact test ((TestStreamCompact::vFunctionCall)shamalgs::numeric::details::stream_compact_fallback);
    test.check();

}