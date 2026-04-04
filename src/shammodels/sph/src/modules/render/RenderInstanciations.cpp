#include "shambase/aliases_float.hpp"
#include "shammodels/common/modules/render/RenderFieldGetter.hpp"
#include "shammodels/common/modules/render/CartesianRender.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/common/modules/render/RenderConfig.hpp"
#include "shammath/sphkernels.hpp"
#include "shampylib/PatchDataToPy.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <pybind11/pytypes.h>

#pragma once

using namespace shammath;

template class shammodels::common::modules::RenderFieldGetter<f64_3, f64, M4, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::RenderFieldGetter<f64_3, f64, M6, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::RenderFieldGetter<f64_3, f64, M8, shammodels::sph::SolverStorage<f64_3, u32>>;

template class shammodels::common::modules::RenderFieldGetter<f64_3, f64, C2, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::RenderFieldGetter<f64_3, f64, C4, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::RenderFieldGetter<f64_3, f64, C6, shammodels::sph::SolverStorage<f64_3, u32>>;

template class shammodels::common::modules::RenderFieldGetter<f64_3, f64_3, M4, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::RenderFieldGetter<f64_3, f64_3, M6, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::RenderFieldGetter<f64_3, f64_3, M8, shammodels::sph::SolverStorage<f64_3, u32>>;

template class shammodels::common::modules::RenderFieldGetter<f64_3, f64_3, C2, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::RenderFieldGetter<f64_3, f64_3, C4, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::RenderFieldGetter<f64_3, f64_3, C6, shammodels::sph::SolverStorage<f64_3, u32>>;

template class shammodels::common::modules::CartesianRender<f64_3, f64, M4, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::CartesianRender<f64_3, f64, M6, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::CartesianRender<f64_3, f64, M8, shammodels::sph::SolverStorage<f64_3, u32>>;

template class shammodels::common::modules::CartesianRender<f64_3, f64, C2, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::CartesianRender<f64_3, f64, C4, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::CartesianRender<f64_3, f64, C6, shammodels::sph::SolverStorage<f64_3, u32>>;

template class shammodels::common::modules::CartesianRender<f64_3, f64_3, M4, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::CartesianRender<f64_3, f64_3, M6, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::CartesianRender<f64_3, f64_3, M8, shammodels::sph::SolverStorage<f64_3, u32>>;

template class shammodels::common::modules::CartesianRender<f64_3, f64_3, C2, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::CartesianRender<f64_3, f64_3, C4, shammodels::sph::SolverStorage<f64_3, u32>>;
template class shammodels::common::modules::CartesianRender<f64_3, f64_3, C6, shammodels::sph::SolverStorage<f64_3, u32>>;

