// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>


inline std::vector<f64> get_h_test_vals(){
    std::vector<f64> ret{};

    for(f64 hfact = 0; hfact < 3; hfact += 0.01){
        ret.push_back(hfact);
    }

    return ret;
}


TestStart(ValidationTest, "shammodels/sph/hfact_default", test_sph_hfact_default, 1) {



}