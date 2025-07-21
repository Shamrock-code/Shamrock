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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/format.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"

// forward declare PatchDataField
template<class T>
class PatchDataField;

namespace shamrock {

    namespace details {

        /// Accessor for read-write access to dynamic nvar buffer data
        template<class T>
        struct PatchDataFieldSpan_access_rw_dyn_nvar {
            T *ptr;   ///< Pointer to the data
            u32 nvar; ///< Number of variables

            /// Access to element at index idx and offset var
            T &operator()(u32 idx, u32 offset) const { return ptr[idx * nvar + offset]; }

            /// Access the underlying pointer
            T &operator[](u32 idx) const { return ptr[idx]; }
        };

        /// Accessor for read-only access to dynamic nvar buffer data
        template<class T>
        struct PatchDataFieldSpan_access_ro_dyn_nvar {
            const T *ptr; ///< Pointer to the data
            u32 nvar;     ///< Number of variables

            /// Access to element at index idx and offset var
            const T &operator()(u32 idx, u32 offset) const { return ptr[idx * nvar + offset]; }

            /// Access the underlying pointer
            const T &operator[](u32 idx) const { return ptr[idx]; }
        };

        /// Accessor for read-write access to static nvar buffer data
        template<class T, u32 nvar>
        struct PatchDataFieldSpan_access_rw_static_nvar {
            T *ptr; ///< Pointer to the data

            /// Access to element at index idx and offset var
            T &operator()(u32 idx, u32 offset) const { return ptr[idx * nvar + offset]; }

            /// Access without offset if nvar is 1
            template<typename Dummy = void, typename = std::enable_if_t<nvar == 1, Dummy>>
            T &operator()(u32 idx) const {
                return ptr[idx];
            }

            /// Access the underlying pointer
            T &operator[](u32 idx) const { return ptr[idx]; }
        };

        /// Accessor for read-only access to static nvar buffer data
        template<class T, u32 nvar>
        struct PatchDataFieldSpan_access_ro_static_nvar {
            const T *ptr; ///< Pointer to the data

            /// Access to element at index idx and offset var
            const T &operator()(u32 idx, u32 offset) const { return ptr[idx * nvar + offset]; }

            /// Access without offset if nvar is 1
            template<typename Dummy = void, typename = std::enable_if_t<nvar == 1, Dummy>>
            const T &operator()(u32 idx) const {
                return ptr[idx];
            }

            /// Access the underlying pointer
            T &operator[](u32 idx) const { return ptr[idx]; }
        };
    } // namespace details

    /// Alias for PatchDataFieldSpan_access_rw_dyn_nvar
    template<class T>
    using pdat_span_rw_dyn = details::PatchDataFieldSpan_access_rw_dyn_nvar<T>;

    /// Alias for PatchDataFieldSpan_access_ro_dyn_nvar
    template<class T>
    using pdat_span_ro_dyn = details::PatchDataFieldSpan_access_ro_dyn_nvar<T>;

    /// Alias for PatchDataFieldSpan_access_rw_static_nvar
    template<class T, u32 nvar>
    using pdat_span_rw = details::PatchDataFieldSpan_access_rw_static_nvar<T, nvar>;

    /// Alias for PatchDataFieldSpan_access_ro_static_nvar
    template<class T, u32 nvar>
    using pdat_span_ro = details::PatchDataFieldSpan_access_ro_static_nvar<T, nvar>;

    /// Constant for dynamic number of variables
    inline constexpr u32 dynamic_nvar = u32_max;

    inline constexpr bool access_t_pointer = true;
    inline constexpr bool access_t_span    = !access_t_pointer;

    /**
     * @class PatchDataFieldSpan
     * @brief Represents a span of data within a PatchDataField.
     *
     * This class provides a way to access a contiguous range of elements within a
     * PatchDataField. It provides either static or dynamic number of variables.
     *
     * @code {.cpp}
     * PatchDataField<T> field("test", 2, cnt_test / 2);
     * field.override(test_vals, cnt_test);
     *
     * shamrock::PatchDataFieldSpan<T, shamrock::dynamic_nvar> span(field, 0, cnt_test);
     *
     * sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());s
     *
     * sham::kernel_call(
     *     shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
     *     sham::MultiRef{span},
     *     sham::MultiRef{ret},
     *     cnt_test / 2,
     *     [](u32 i, auto sp, T *ret_val) {
     *         ret_val[i * 2 + 0] = sp(i, 0);
     *         ret_val[i * 2 + 1] = sp(i, 1);
     *     });
     * @endcode
     *
     * @tparam T The type of data stored in the PatchDataField.
     * @tparam nvar The number of variables in the PatchDataField. Defaults to
     * dynamic_nvar.
     */
    template<class T, u32 nvar = dynamic_nvar, bool pointer_access = access_t_span>
    class PatchDataFieldSpan {
        public:
        /**
         * @brief Returns true if the number of variables is dynamic.
         *
         * @return True if the number of variables is dynamic, false otherwise.
         */
        inline static constexpr bool is_nvar_dynamic() { return nvar == dynamic_nvar; }

        /**
         * @brief Returns true if the number of variables is static.
         *
         * @return True if the number of variables is static, false otherwise.
         */
        inline static constexpr bool is_nvar_static() { return nvar != dynamic_nvar; }

        inline static constexpr bool is_pointer_access() {
            return pointer_access == access_t_pointer;
        }
        inline static constexpr bool is_span_access() { return pointer_access == access_t_span; }

        /**
         * @brief Constructor.
         *
         * Initializes the span with a reference to a PatchDataField, a starting index,
         * and a count of elements.
         *
         * @param field_ref Reference to the PatchDataField.
         * @param start Starting index of the span.
         * @param count Number of elements in the span.
         *
         * @throws std::invalid_argument If the underlying buffer is empty.
         * @throws std::invalid_argument If the number of variables is static and does
         * not match the number of variables in the PatchDataField.
         */
        PatchDataFieldSpan(
            PatchDataField<T> &field_ref,
            u32 start,
            u32 count,
            SourceLocation loc = SourceLocation{})
            : field_ref(field_ref), start(start), count(count) {

            StackEntry stack_loc{};

            // ensure that the underlying USM pointer can be accessed
            if (field_ref.get_buf().is_empty()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "PatchDataFieldSpan can not be binded to empty buffer", loc);
            }

            if (is_nvar_static()) {
                if (field_ref.get_nvar() != nvar) {
                    shambase::throw_with_loc<std::invalid_argument>(
                        shambase::format(
                            "You are trying to bind a PatchDataFieldSpan with static nvar={} to a "
                            "PatchDataField with nvar={}",
                            nvar,
                            field_ref.get_nvar()),
                        loc);
                }
            }

            if (start + count > field_ref.get_obj_cnt()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format(
                        "PatchDataFieldSpan out of bounds: {} + {} > {}",
                        start,
                        count,
                        field_ref.get_obj_cnt()),
                    loc);
            }
        }

        /**
         * @brief Returns a read-only accessor to the data in the span.
         *
         * The type of accessor returned depends on whether the number of variables is
         * static or dynamic.
         *
         * @param depends_list Event list that the accessor depends on.
         *
         * @return Read-only accessor to the data in the span.
         */
        template<
            typename Dummy = void,
            typename       = std::enable_if_t<is_nvar_dynamic() && is_span_access(), Dummy>>
        inline auto get_read_access(
            sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{}) const
            -> details::PatchDataFieldSpan_access_ro_dyn_nvar<T> {
            StackEntry stack_loc{};
            return details::PatchDataFieldSpan_access_ro_dyn_nvar<T>{
                get_buf().get_read_access(depends_list, std::move(src_loc))
                    + start * field_ref.get_nvar(),
                field_ref.get_nvar()};
        }

        /**
         * @brief Returns a read-write accessor to the data in the span.
         *
         * The type of accessor returned depends on whether the number of variables is
         * static or dynamic.
         *
         * @param depends_list Event list that the accessor depends on.
         *
         * @return Read-write accessor to the data in the span.
         */
        template<
            typename Dummy = void,
            typename       = std::enable_if_t<is_nvar_dynamic() && is_span_access(), Dummy>>
        inline auto
        get_write_access(sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{})
            -> details::PatchDataFieldSpan_access_rw_dyn_nvar<T> {
            StackEntry stack_loc{};
            return details::PatchDataFieldSpan_access_rw_dyn_nvar<T>{
                get_buf().get_write_access(depends_list, std::move(src_loc))
                    + start * field_ref.get_nvar(),
                field_ref.get_nvar()};
        }

        /**
         * @brief Returns a read-only accessor to the data in the span.
         *
         * The type of accessor returned depends on whether the number of variables is
         * static or dynamic.
         *
         * @param depends_list Event list that the accessor depends on.
         *
         * @return Read-only accessor to the data in the span.
         */
        template<
            typename Dummy = void,
            typename       = std::enable_if_t<is_nvar_static() && is_span_access(), Dummy>>
        inline auto get_read_access(
            sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{}) const
            -> details::PatchDataFieldSpan_access_ro_static_nvar<T, nvar> {
            StackEntry stack_loc{};
            return details::PatchDataFieldSpan_access_ro_static_nvar<T, nvar>{
                get_buf().get_read_access(depends_list, std::move(src_loc))
                + start * field_ref.get_nvar()};
        }

        /**
         * @brief Returns a read-write accessor to the data in the span.
         *
         * The type of accessor returned depends on whether the number of variables is
         * static or dynamic.
         *
         * @param depends_list Event list that the accessor depends on.
         *
         * @return Read-write accessor to the data in the span.
         */
        template<
            typename Dummy = void,
            typename       = std::enable_if_t<is_nvar_static() && is_span_access(), Dummy>>
        inline auto
        get_write_access(sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{})
            -> details::PatchDataFieldSpan_access_rw_static_nvar<T, nvar> {
            StackEntry stack_loc{};
            return details::PatchDataFieldSpan_access_rw_static_nvar<T, nvar>{
                get_buf().get_write_access(depends_list, std::move(src_loc))
                + start * field_ref.get_nvar()};
        }

        template<typename Dummy = void, typename = std::enable_if_t<is_pointer_access(), Dummy>>
        inline auto get_read_access(
            sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{}) const
            -> const T * {
            StackEntry stack_loc{};
            return {
                get_buf().get_read_access(depends_list, std::move(src_loc))
                + start * field_ref.get_nvar()};
        }

        template<typename Dummy = void, typename = std::enable_if_t<is_pointer_access(), Dummy>>
        inline auto
        get_write_access(sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{})
            -> T * {
            StackEntry stack_loc{};
            return {
                get_buf().get_write_access(depends_list, std::move(src_loc))
                + start * field_ref.get_nvar()};
        }

        /**
         * @brief Completes the event state of the underlying buffer.
         *
         * @param e Event to complete.
         */
        inline void complete_event_state(sycl::event e) const {
            StackEntry stack_loc{};
            get_buf().complete_event_state(e);
        }

        /// Reference to the PatchDataField.
        PatchDataField<T> &field_ref;

        /// Starting element index of the span.
        u32 start;

        /// Number of elements
        u32 count;

        /**
         * @brief Copy the current span to a new DeviceBuffer
         *
         * @return A new DeviceBuffer containing the data from the current span.
         */
        inline sham::DeviceBuffer<T> to_buf() const {
            auto dev_sched = get_buf().get_dev_scheduler_ptr();
            auto ncopy     = count * field_ref.get_nvar();
            sham::DeviceBuffer<T> buf(ncopy, dev_sched);

            // option 1
            // sham::kernel_call(
            //    dev_sched->get_queue(),
            //    sham::MultiRef{*this},
            //    sham::MultiRef{buf},
            //    ncopy,
            //    [](u32 i, auto indata, T *outdata) {
            //        outdata[i] = indata[i];
            //    });

            // option 2
            sham::EventList depends_list;
            auto ptr_source
                = get_buf().get_read_access(depends_list) + start * field_ref.get_nvar();

            auto ptr_target = buf.get_write_access(depends_list);

            sycl::event e = dev_sched->get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                cgh.copy(ptr_source, ptr_target, ncopy);
            });

            get_buf().complete_event_state(e);
            buf.complete_event_state(e);

            return buf;
        }

        private:
        /// Returns the underlying buffer of the PatchDataField.
        inline sham::DeviceBuffer<T> &get_buf() { return field_ref.get_buf(); }

        /// const variant of get_buf
        inline const sham::DeviceBuffer<T> &get_buf() const { return field_ref.get_buf(); }
    };

    template<class T>
    using PatchDataFieldSpanPointer = PatchDataFieldSpan<T, dynamic_nvar, access_t_pointer>;

} // namespace shamrock
