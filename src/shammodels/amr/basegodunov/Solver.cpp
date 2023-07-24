// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/basegodunov/Solver.hpp"

template<class Tvec, class TgridVec>
using Solver = shammodels::basegodunov::Solver<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
auto Solver<Tvec, TgridVec>::evolve_once(Tscal t_current, Tscal dt_input) -> Tscal{

    //ghost zone exchange
    
    //compute bound received

    //round to next pow of 2

    //build radix trees

    //build neigh table
}