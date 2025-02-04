// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchDataFieldSpan.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/format.hpp"
#include "shamrock/patch/PatchDataField.hpp"

namespace shamrock {

    namespace details {

        template<class T>
        struct PatchDataFieldSpan_access_rw_dyn_nvar {
            T *ptr;
            u32 nvar;

            T &operator()(u32 idx, u32 offset) const { return ptr[idx * nvar + offset]; }
        };

        template<class T>
        struct PatchDataFieldSpan_access_ro_dyn_nvar {
            const T *ptr;
            u32 nvar;

            const T &operator()(u32 idx, u32 offset) const { return ptr[idx * nvar + offset]; }
        };
        template<class T, u32 nvar>
        struct PatchDataFieldSpan_access_rw_static_nvar {
            T *ptr;

            T &operator()(u32 idx, u32 offset) const { return ptr[idx * nvar + offset]; }

            // same operator but without offset since nvar == 1, enable only if nvar is 1
            template<std::enable_if_t<nvar == 1, int> = 0>
            T &operator()(u32 idx) const {
                return ptr[idx];
            }
        };

        template<class T, u32 nvar>
        struct PatchDataFieldSpan_access_ro_static_nvar {
            const T *ptr;

            const T &operator()(u32 idx, u32 offset) const { return ptr[idx * nvar + offset]; }

            // same operator but without offset since nvar == 1, enable only if nvar is 1
            template<std::enable_if_t<nvar == 1, int> = 0>
            T &operator()(u32 idx) const {
                return ptr[idx];
            }
        };
    } // namespace details

    inline constexpr u32 dynamic_nvar = -1;

    template<class T, u32 nvar = dynamic_nvar>
    class PatchDataFieldSpan {

        inline static constexpr bool is_nvar_dynamic() { return nvar == dynamic_nvar; }
        inline static constexpr bool is_nvar_static() { return nvar != dynamic_nvar; }

        PatchDataField<T> &field_ref;
        u32 start, count;

        PatchDataFieldSpan(PatchDataField<T> &field_ref, u32 start, u32 count)
            : field_ref(field_ref), start(start), count(count) {

            // ensure that the underlying USM pointer can be accessed
            if (field_ref.get_buf().is_empty()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "PatchDataFieldSpan can not be binded to empty buffer");
            }

            if (is_nvar_static()) {
                if (field_ref.get_nvar() != nvar) {
                    shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                        "You are trying to bind a PatchDataFieldSpan with static nvar={} to a "
                        "PatchDataField with nvar={}",
                        nvar,
                        field_ref.get_nvar()));
                }
            }
        }

        template<std::enable_if_t<is_nvar_dynamic(), int> = 0>
        inline auto get_read_access(sham::EventList &depends_list)
            -> details::PatchDataFieldSpan_access_ro_dyn_nvar<T> {
            return details::PatchDataFieldSpan_access_ro_dyn_nvar<T>{
                field_ref.get_read_access(depends_list), field_ref.get_nvar()};
        }

        template<std::enable_if_t<is_nvar_static(), int> = 0>
        inline auto get_read_access(sham::EventList &depends_list)
            -> details::PatchDataFieldSpan_access_ro_static_nvar<T, nvar> {
            return details::PatchDataFieldSpan_access_ro_static_nvar<T, nvar>{
                field_ref.get_read_access(depends_list)};
        }

        template<std::enable_if_t<is_nvar_dynamic(), int> = 0>
        inline auto get_write_access(sham::EventList &depends_list)
            -> details::PatchDataFieldSpan_access_rw_dyn_nvar<T> {
            return details::PatchDataFieldSpan_access_rw_dyn_nvar<T>{
                field_ref.get_write_access(depends_list), field_ref.get_nvar()};
        }

        template<std::enable_if_t<is_nvar_static(), int> = 0>
        inline auto get_write_access(sham::EventList &depends_list)
            -> details::PatchDataFieldSpan_access_rw_static_nvar<T, nvar> {
            return details::PatchDataFieldSpan_access_rw_static_nvar<T, nvar>{
                field_ref.get_write_access(depends_list)};
        }
    };

} // namespace shamrock
