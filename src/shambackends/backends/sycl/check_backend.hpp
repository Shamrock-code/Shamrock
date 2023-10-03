// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#ifndef SHAMROCK_ENABLE_BACKEND_SYCL
#error You have included a file belonging to the sycl backend, but you are not compiling with the sycl backend
#endif