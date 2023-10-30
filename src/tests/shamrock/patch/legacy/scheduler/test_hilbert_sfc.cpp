// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/shamtest.hpp"

#include "shamrock/sfc/hilbert.hpp"
#if false

Test_start("sfc::", hilbert, 1) {
    
    Test_assert("compute_hilbert_index(5, 10, 20, 5) == 7865", compute_hilbert_index_3d<5>(5, 10, 20) == 7865);
    Test_assert("compute_hilbert_index(0,0,0, 21) == 7865", compute_hilbert_index_3d<21>(0,0,0) == 0);
    Test_assert("compute_hilbert_index(hilbert_box21_sz,0,0, 21)", compute_hilbert_index_3d<21>(hilbert_box21_sz,0,0) == 9223372036854775807);

}
#endif