// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchDataLayerLayout.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamrock/patch/FieldVariant.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <variant>
#include <vector>

namespace shamrock::patch {

    /**
     * @brief Structure describing a field in a patch data layout.
     *
     * The field descriptor contains the name of the field and the number of
     * variables of the field.
     */
    template<class T>
    class FieldDescriptor {
        public:
        /// The type of the field mirrored from the template type
        using field_T = T;

        /// The name of the field
        std::string name;

        /// The number of variables of the field per object
        u32 nvar;

        /**
         * @brief Default constructor.
         *
         * The default constructor initializes the field descriptor with an
         * empty name and 1 variable.
         */
        inline FieldDescriptor() : name(""), nvar(1) {};

        /**
         * @brief Constructor with a given name and number of variables.
         *
         * @param name The name of the field.
         * @param nvar The number of variables of the field.
         */
        inline FieldDescriptor(std::string name, u32 nvar) : nvar(nvar), name(name) {}
    };

    class PatchDataLayerLayout {

        template<class T>
        using FieldDescriptor = shamrock::patch::FieldDescriptor<T>;

        // using var_t = var_t_template<FieldDescriptor>;
        using var_t = FieldVariant<FieldDescriptor>;

        std::vector<var_t> fields;

        public:
        /**
         * @brief add a field of type T to the layout
         *
         * @tparam T type of the field
         * @param field_name field name
         * @param nvar number of varaible per object
         */
        template<class T>
        void add_field(std::string field_name, u32 nvar, SourceLocation loc = SourceLocation{});

        /**
         * @brief Get the field description id if matching name & type
         *
         * @tparam T
         * @param field_name
         * @return FieldDescriptor<T>
         */
        template<class T>
        FieldDescriptor<T> get_field(std::string field_name);

        /**
         * @brief Get the field description at given index
         *
         * @tparam T
         * @param idx
         * @return FieldDescriptor<T>
         */
        template<class T>
        FieldDescriptor<T> get_field(u32 idx);

        /**
         * @brief Get the field id if matching name & type
         *
         * @tparam T
         * @param field_name
         * @return u32
         */
        template<class T>
        u32 get_field_idx(std::string field_name);

        /**
         * @brief Get the field id if matching name & type & nvar
         *
         * @tparam T
         * @param field_name
         * @param nvar
         * @return u32
         */
        template<class T>
        u32 get_field_idx(std::string field_name, u32 nvar);

        /**
         * @brief check that field of id @idx is of type T
         *
         * @tparam T
         * @param idx
         * @return true
         * @return false
         */
        template<class T>
        bool check_field_type(u32 idx);

        /**
         * @brief check that main field (id=0)is of type T
         *
         * @tparam T
         * @return true
         * @return false
         */
        template<class T>
        inline bool check_main_field_type() {
            return check_field_type<T>(0);
        }

        /**
         * @brief Get the main field description as a variant object
         *
         * @return const var_t& the variant field description
         */
        [[nodiscard]] inline const var_t &get_main_field_any() const { return fields[0]; }

        /**
         * @brief Get the description of the layout
         *
         * @return std::string
         */
        std::string get_description_str();

        /**
         * @brief Get the list of field names
         *
         * @return std::vector<std::string>
         */
        std::vector<std::string> get_field_names();

        /**
         * @brief for each visit of each field
         *
         * @tparam Functor the signature of the lambda must be : [?](auto & arg){...}
         * @param func
         */
        template<class Functor>
        inline void for_each_field_any(Functor &&func) const {
            for (auto &f : fields) {
                f.visit([&](auto &arg) {
                    func(arg);
                });
            }
        }

        /**
         * @brief Add a field with type specified as a string
         *
         * @param fname the name of the field
         * @param nvar the number of variables
         * @param type the type of the field as a string
         */
        inline void add_field_t(std::string fname, u32 nvar, std::string type) {
            if (type == "f32") {
                add_field<f32>(fname, nvar);
            } else if (type == "f32_2") {
                add_field<f32_2>(fname, nvar);
            } else if (type == "f32_3") {
                add_field<f32_3>(fname, nvar);
            } else if (type == "f32_4") {
                add_field<f32_4>(fname, nvar);
            } else if (type == "f32_8") {
                add_field<f32_8>(fname, nvar);
            } else if (type == "f32_16") {
                add_field<f32_16>(fname, nvar);
            } else if (type == "f64") {
                add_field<f64>(fname, nvar);
            } else if (type == "f64_2") {
                add_field<f64_2>(fname, nvar);
            } else if (type == "f64_3") {
                add_field<f64_3>(fname, nvar);
            } else if (type == "f64_4") {
                add_field<f64_4>(fname, nvar);
            } else if (type == "f64_8") {
                add_field<f64_8>(fname, nvar);
            } else if (type == "f64_16") {
                add_field<f64_16>(fname, nvar);
            } else if (type == "u32") {
                add_field<u32>(fname, nvar);
            } else if (type == "u64") {
                add_field<u64>(fname, nvar);
            } else if (type == "u32_3") {
                add_field<u32_3>(fname, nvar);
            } else if (type == "u64_3") {
                add_field<u64_3>(fname, nvar);
            } else if (type == "i64_3") {
                add_field<i64_3>(fname, nvar);
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "the select type is not recognized");
            }
        }

        /**
         * @brief Overloaded equality operator for PatchDataLayerLayout class
         *
         * This operator is used to check if two PatchDataLayerLayout objects are equal.
         * It compares the fields of the objects and returns true if they are equal,
         * and false otherwise.
         *
         * @param lhs The first PatchDataLayerLayout object to compare
         * @param rhs The second PatchDataLayerLayout object to compare
         *
         * @return true if the two objects are equal, false otherwise
         */
        friend bool operator==(const PatchDataLayerLayout &lhs, const PatchDataLayerLayout &rhs);
    };

    /**
     * @brief Serialize a PatchDataLayerLayout object to a JSON object
     *
     * This function takes a PatchDataLayerLayout object and serializes it to a JSON object.
     * It is used to convert the PatchDataLayerLayout object to a JSON string.
     *
     * @param j The JSON object to serialize the PatchDataLayerLayout object to
     * @param p The PatchDataLayerLayout object to serialize
     */
    void to_json(nlohmann::json &j, const PatchDataLayerLayout &p);

    /**
     * @brief Deserialize a PatchDataLayerLayout object from a JSON object
     *
     * This function takes a JSON object and deserializes it to a PatchDataLayerLayout object.
     * It is used to convert a JSON string to a PatchDataLayerLayout object.
     *
     * @param j The JSON object to deserialize the PatchDataLayerLayout object from
     * @param p The PatchDataLayerLayout object to deserialize
     */
    void from_json(const nlohmann::json &j, PatchDataLayerLayout &p);

    /**
     * @brief Overloaded equality operator for PatchDataLayerLayout class
     *
     * This operator is used to check if two PatchDataLayerLayout objects are equal.
     * It compares the fields of the objects and returns true if they are equal,
     * and false otherwise.
     *
     * @param lhs The first PatchDataLayerLayout object to compare
     * @param rhs The second PatchDataLayerLayout object to compare
     *
     * @return true if the two objects are equal, false otherwise
     */
    bool operator==(const PatchDataLayerLayout &lhs, const PatchDataLayerLayout &rhs);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // out of line implementation of the PatchDataLayerLayout
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline void
    PatchDataLayerLayout::add_field(std::string field_name, u32 nvar, SourceLocation loc) {
        bool found = false;

        for (var_t &fvar : fields) {
            fvar.visit([&](auto &arg) {
                if (field_name == arg.name) {
                    found = true;
                }
            });
        }

        if (found) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "add_field -> the name already exists");
        }

        shamlog_debug_ln(
            "PatchDataLayerLayout",
            "adding field :",
            field_name,
            nvar,
            "loc :",
            loc.format_one_line());

        fields.push_back(var_t{FieldDescriptor<T>(field_name, nvar)});
    }

    template<class T>
    inline PatchDataLayerLayout::FieldDescriptor<T>
    PatchDataLayerLayout::get_field(std::string field_name) {

        for (var_t &fvar : fields) {
            if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fvar.value)) {
                if (pval->name == field_name) {
                    return *pval;
                }
            }
        }

        throw shambase::make_except_with_loc<std::invalid_argument>(
            "the requested field does not exists\n    current table : " + get_description_str());
    }

    template<class T>
    inline PatchDataLayerLayout::FieldDescriptor<T> PatchDataLayerLayout::get_field(u32 idx) {

        if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fields[idx].value)) {
            return *pval;
        }

        throw shambase::make_except_with_loc<std::invalid_argument>(
            "the required type does no match at index " + std::to_string(idx)
            + "\n    current table : " + get_description_str());
    }

    template<class T>
    inline u32 PatchDataLayerLayout::get_field_idx(std::string field_name) {
        for (u32 i = 0; i < fields.size(); i++) {
            if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fields[i].value)) {
                if (pval->name == field_name) {
                    return i;
                }
            }
        }

        throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
            "the requested field does not exists\n    the function : {}\n    the field name : {}\n "
            "   current table : \n{}",
            __PRETTY_FUNCTION__,
            field_name,
            get_description_str()));
    }

    template<class T>
    inline u32 PatchDataLayerLayout::get_field_idx(std::string field_name, u32 nvar) {
        for (u32 i = 0; i < fields.size(); i++) {
            if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fields[i].value)) {
                if ((pval->name == field_name) && (pval->nvar == nvar)) {
                    return i;
                }
            }
        }

        throw shambase::make_except_with_loc<std::invalid_argument>(
            "the requested field does not exists\n    current table : " + get_description_str());
    }

    template<class T>
    inline bool PatchDataLayerLayout::check_field_type(u32 idx) {
        var_t &tmp = fields[idx];

        FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&tmp.value);

        if (pval) {
            return true;
        } else {
            return false;
        }
    }

} // namespace shamrock::patch
