// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyPatchDataLazyGetter.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shamrock::PatchDataLazyGetter
 */

#include "shambindings/pybindaliases.hpp"
#include "shampylib/PatchDataToPy.hpp"
#include "sph_pybindings.hpp"

namespace shammodels::sph::pybindings {

    void add_patch_data_lazy_getter(py::module &m) {
        py::class_<shamrock::PatchDataLazyGetter>(m, "PatchDataLazyGetter")
            .def("__getitem__", &shamrock::PatchDataLazyGetter::get_item);
    }

} // namespace shammodels::sph::pybindings
