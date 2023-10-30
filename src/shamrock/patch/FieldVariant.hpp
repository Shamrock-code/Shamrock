// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file FieldVariant.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */


#include "shambackends/typeAliasVec.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambase/exception.hpp"
#include <variant>

namespace shamrock::patch {

    /**
     * @brief Field variant class
     * \todo provide exemple
     * @tparam Container 
     */
    template<template<class> class Container>
    class FieldVariant {

        public:
        using var_t_template = std::variant<
            Container<f32>,
            Container<f32_2>,
            Container<f32_3>,
            Container<f32_4>,
            Container<f32_8>,
            Container<f32_16>,
            Container<f64>,
            Container<f64_2>,
            Container<f64_3>,
            Container<f64_4>,
            Container<f64_8>,
            Container<f64_16>,
            Container<u32>,
            Container<u64>,
            Container<u32_3>,
            Container<u64_3>,
            Container<i64_3>
            >;

        var_t_template value; /*!< The held variant */

        template<class T>
        explicit FieldVariant(Container<T> && val) : value(std::move(val)) {}

        template<class T>
        Container<T> &get_if_ref_throw() {
            if (Container<T> *pval = std::get_if<Container<T>>(&value)) {
                return *pval;
            }
            throw shambase::throw_with_loc<std::invalid_argument>("the type asked is not correct");
        }

        template<class Func>
        void visit(Func && f){
            std::visit([&](auto &arg) { f(arg); }, value);
        }

        template<class Func>
        auto visit_return(Func && f){
            return std::visit([&](auto &arg) { return f(arg); }, value);
        }

        template<class Func>
        void visit(Func && f) const {
            std::visit([&](auto &arg) { f(arg); }, value);
        }

        template<template<class> class Container2, class Func> 
        FieldVariant<Container2> convert(Func && f){
            return std::visit([&](auto &arg) { return f(arg); }, value);
        }
    };

    template<template<class> class Container>
    using var_t_template = typename FieldVariant<Container>::var_t_template;

} // namespace shamrock::patch