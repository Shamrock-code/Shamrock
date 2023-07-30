// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include <memory>

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/amr/basegodunov/Model.hpp"


const std::string base_name = "AMRGodunov";

template<class Tvec, class TgridVec>
void add_instance(py::module &m, std::string name_config, std::string name_model){

    using namespace shammodels::basegodunov;

    using Tscal              = shambase::VecComponent<Tvec>;
    using Tgridscal          = shambase::VecComponent<TgridVec>;

    using T       = Model<Tvec, TgridVec>;
    using TConfig = typename T::Solver::Config;


    py::class_<TConfig>(m, name_config.c_str());

    py::class_<T>(m, name_model.c_str())
    .def("init_scheduler",&T::init_scheduler)
    .def("make_base_grid",&T::make_base_grid)
    .def("dump_vtk",&T::dump_vtk)
    .def("evolve_once",&T::evolve_once);

}

Register_pymod(pybasegodunovmodel) {

    using namespace shammodels::basegodunov;

    add_instance<f64_3, i64_3>(
        m, base_name + "_f64_3_i64_3_SolverConfig", base_name + "_f64_3_i64_3_Model");

    using VariantAMRGodunovBind =
        std::variant<std::unique_ptr<Model<f64_3, i64_3>>>;

    m.def(
        "get_AMRGodunov",
        [](ShamrockCtx &ctx, std::string vector_type, std::string grid_repr) -> VariantAMRGodunovBind {
            VariantAMRGodunovBind ret;

            if (vector_type == "f64_3" && grid_repr == "i64_3") {
                ret = std::make_unique<Model<f64_3, i64_3>>(ctx);
            } else {
                throw shambase::throw_with_loc<std::invalid_argument>(
                    "unknown combination of representation and grid_repr");
            }

            return ret;
        },
        py::kw_only(),
        py::arg("context"),
        py::arg("vector_type"),
        py::arg("grid_repr"));
}