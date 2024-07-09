// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeEos.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/AnalysisSodTube.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisSodTube<Tvec, SPHKernel>::compute_L2_dist() -> field_val{

    





    return field_val{1,Tvec{0,1,0},1};

}

using namespace shammath;
template class shammodels::sph::modules::AnalysisSodTube<f64_3, M4>;
template class shammodels::sph::modules::AnalysisSodTube<f64_3, M6>;
template class shammodels::sph::modules::AnalysisSodTube<f64_3, M8>;