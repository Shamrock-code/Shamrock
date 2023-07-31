// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamrock/patch/FieldVariant.hpp"
#include "shamsys/legacy/log.hpp"
#include <sstream>
#include <variant>
#include <vector>

namespace shamrock::patch {
    class PatchDataLayout {

        template<class T>
        class FieldDescriptor {
            public:
            using field_T = T;

            std::string name;
            u32 nvar;

            inline FieldDescriptor() : name(""), nvar(1){};
            inline FieldDescriptor(std::string name, u32 nvar) : nvar(nvar), name(name) {}
        };
        

        //using var_t = var_t_template<FieldDescriptor>;
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
        u32 get_field_idx(std::string field_name,u32 nvar);

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
        inline void for_each_field_any(Functor &&func) {
            for (auto &f : fields) {
                f.visit([&](auto &arg) { func(arg); });
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // out of line implementation of the PatchDataLayout
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline void PatchDataLayout::add_field(std::string field_name, u32 nvar, SourceLocation loc) {
        bool found = false;

        for (var_t &fvar : fields) {
            fvar.visit(
                [&](auto &arg) {
                    if (field_name == arg.name) {
                        found = true;
                    }
                }
            );
        }

        if (found) {
            throw shambase::throw_with_loc<std::invalid_argument>("add_field -> the name already exists");
        }

        logger::debug_ln("PatchDataLayout", "adding field :", field_name, nvar, "loc :",loc.format_one_line());

        fields.push_back(var_t{FieldDescriptor<T>(field_name, nvar)});
    }

    template<class T>
    inline PatchDataLayout::FieldDescriptor<T> PatchDataLayout::get_field(std::string field_name) {

        for (var_t &fvar : fields) {
            if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fvar.value)) {
                if (pval->name == field_name) {
                    return *pval;
                }
            }
        }

        throw shambase::throw_with_loc<std::invalid_argument>(
            "the requested field does not exists\n    current table : " + get_description_str()
        );
    }

    template<class T>
    inline PatchDataLayout::FieldDescriptor<T> PatchDataLayout::get_field(u32 idx) {

        if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fields[idx].value)) {
            return *pval;
        }

        throw shambase::throw_with_loc<std::invalid_argument>(
            "the required type does no match at index "+std::to_string(idx)+"\n    current table : " + get_description_str()
        );
    }

    template<class T>
    inline u32 PatchDataLayout::get_field_idx(std::string field_name) {
        for (u32 i = 0; i < fields.size(); i++) {
            if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fields[i].value)) {
                if (pval->name == field_name) {
                    return i;
                }
            }
        }

        throw shambase::throw_with_loc<std::invalid_argument>(
            shambase::format("the requested field does not exists\n    the function : {}\n    the field name : {}\n    current table : \n{}", __PRETTY_FUNCTION__, field_name, get_description_str())
        );
    }

    template<class T>
    inline u32 PatchDataLayout::get_field_idx(std::string field_name,u32 nvar) {
        for (u32 i = 0; i < fields.size(); i++) {
            if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fields[i].value)) {
                if ((pval->name == field_name) && (pval->nvar == nvar)) {
                    return i;
                }
            }
        }

        throw shambase::throw_with_loc<std::invalid_argument>(
            "the requested field does not exists\n    current table : " + get_description_str()
        );
    }

    template<class T>
    inline bool PatchDataLayout::check_field_type(u32 idx) {
        var_t &tmp = fields[idx];

        FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&tmp.value);

        if (pval) {
            return true;
        } else {
            return false;
        }
    }

} // namespace shamrock::patch