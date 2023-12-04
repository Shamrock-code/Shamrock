// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeLoadBalanceValue.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "ComputeLoadBalanceValue.hpp"
#include "shamsys/legacy/log.hpp"
#include "shammath/sphkernels.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ComputeLoadBalanceValue<Tvec, SPHKernel>::update_load_balancing() {

    logger::debug_ln("ComputeLoadBalanceValue", "update load balancing");
    scheduler().update_local_load_value([&](shamrock::patch::Patch p){
        return scheduler().patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });

}

using namespace shammath;
template class shammodels::sph::modules::ComputeLoadBalanceValue<f64_3, M4>;
template class shammodels::sph::modules::ComputeLoadBalanceValue<f64_3, M6>;