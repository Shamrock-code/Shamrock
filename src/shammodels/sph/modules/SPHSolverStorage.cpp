// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "SPHSolverStorage.hpp"

template<class Tvec>
using Storage = shammodels::SPHSolverStorage<Tvec>;


template class shammodels::SPHSolverStorage<f64_3>;