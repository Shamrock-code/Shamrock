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
 * @file kernel_call.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/optional.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include <functional>
#include <optional>

namespace sham {

    namespace details {

        /**
         * @brief Get a pointer to the data of an optional device buffer, for reading.
         * @details If the optional is empty, a null pointer is returned. Otherwise, the read
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param buffer An optional holding a reference to the device buffer.
         * @param depends_list The list of events to wait for.
         * @return A pointer to the data of the buffer, or nullptr if the optional is empty.
         */
        template<class T>
        const T *read_access_optional(
            shambase::opt_ref<sham::DeviceBuffer<T>> buffer, sham::EventList &depends_list) {
            if (!buffer.has_value()) {
                return nullptr;
            } else {
                return buffer.value().get().get_read_access(depends_list);
            }
        }

        /**
         * @brief Get a pointer to the data of an optional device buffer, for writing.
         * @details If the optional is empty, a null pointer is returned. Otherwise, the write
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param buffer An optional holding a reference to the device buffer.
         * @param depends_list The list of events to wait for.
         * @return A pointer to the data of the buffer, or nullptr if the optional is empty.
         */
        template<class T>
        T *write_access_optional(
            shambase::opt_ref<sham::DeviceBuffer<T>> buffer, sham::EventList &depends_list) {
            if (!buffer.has_value()) {
                return nullptr;
            } else {
                return buffer.value().get().get_write_access(depends_list);
            }
        }

        /**
         * @brief Complete the event state of an optional device buffer.
         * @details If the optional is empty, nothing is done. Otherwise, the event state of the
         * buffer is completed with the given event.
         */
        template<class T>
        void complete_state_optional(sycl::event e, shambase::opt_ref<T> buffer) {
            if (buffer.has_value()) {
                buffer.value().get().complete_event_state(e);
            }
        }

        template<class Obj>
        inline auto get_read_access(Obj &o, sham::EventList &depends_list) {
            return o.get_read_access(depends_list);
        }

        template<class Obj>
        inline auto get_write_access(Obj &o, sham::EventList &depends_list) {
            return o.get_write_access(depends_list);
        }
        template<class Obj>
        inline auto complete_event_state(Obj &o, sycl::event e) {
            return o.complete_event_state(e);
        }

        template<class Obj>
        inline auto get_read_access(std::reference_wrapper<Obj> &o, sham::EventList &depends_list) {
            return o.get().get_read_access(depends_list);
        }

        template<class Obj>
        inline auto get_write_access(
            std::reference_wrapper<Obj> &o, sham::EventList &depends_list) {
            return o.get().get_write_access(depends_list);
        }
        template<class Obj>
        inline auto complete_event_state(std::reference_wrapper<Obj> &o, sycl::event e) {
            return o.get().complete_event_state(e);
        }

    } // namespace details

    /**
     * @brief Converts a reference to a given object into an optional reference wrapper.
     * @tparam T Type of the object to reference.
     * @param t Reference to the object.
     * @return An std::optional containing a std::reference_wrapper of the object.
     */
    template<class T>
    shambase::opt_ref<T> to_opt_ref(T &t) {
        return t;
    }

    /**
     * @brief Returns an empty optional containing a reference to a sham::DeviceBuffer<T>.
     * @details This function is useful when you want to pass an optional reference to a kernel
     * argument but you don't know if the argument is going to be used or not.
     * @return An empty std::optional containing a std::reference_wrapper of a
     * sham::DeviceBuffer<T>.
     */
    template<class T>
    auto empty_buf_ref() {
        return shambase::opt_ref<sham::DeviceBuffer<T>>{};
    }

    /**
     * @brief A variant of MultiRef for optional buffers.
     *
     * This class is equivalent to MultiRef but it allows optional buffers. Only DeviceBuffer are
     * supported as optional buffers.
     *
     * @see MultiRef
     */
    template<class... Targ>
    struct MultiRefOpt {
        /// A tuple of optional references to the buffers.
        using storage_t = std::tuple<shambase::opt_ref<Targ>...>;

        /// The tuple of optional references to the buffers.
        storage_t storage;

        /// Constructor from a tuple of optional references to the buffers.
        MultiRefOpt(shambase::opt_ref<Targ>... arg) : storage(arg...) {}

        /**
         * @brief Get a tuple of pointers to the data of the buffers, for reading.
         * @details If a buffer is empty, a null pointer is returned. Otherwise, the read
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param depends_list The list of events to wait for.
         * @return A tuple of pointers to the data of the buffers, or nullptr if the buffer is
         * empty.
         */
        auto get_read_access(sham::EventList &depends_list) {
            __shamrock_stack_entry();
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::read_access_optional(__a, depends_list)...);
                },
                storage);
        }
        /**
         * @brief Get a tuple of pointers to the data of the buffers, for writing.
         * @details If a buffer is empty, a null pointer is returned. Otherwise, the write
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param depends_list The list of events to wait for.
         * @return A tuple of pointers to the data of the buffers, or nullptr if the buffer is
         * empty.
         */
        auto get_write_access(sham::EventList &depends_list) {
            __shamrock_stack_entry();
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::write_access_optional(__a, depends_list)...);
                },
                storage);
        }

        /**
         * @brief Complete the event state of the buffers.
         * @details This function completes the event state of all the buffers in the
         * MultiRefOpt by registering the event `e` in all the buffers.
         *
         * @param e The SYCL event to register in the buffers.
         */
        void complete_event_state(sycl::event e) {
            __shamrock_stack_entry();
            std::apply(
                [&](auto &...__in) {
                    ((details::complete_state_optional(e, __in)), ...);
                },
                storage);
        }
    };

    namespace details {
        /// internal_utility for MultiRef template deduction guide
        template<class T>
        struct mapper {
            /// The mapped type.
            using type = T;
        };

        /// internal_utility for MultiRef template deduction guide
        template<class T>
        struct mapper<shambase::opt_ref<T>> {
            /// The mapped type.
            using type = T;
        };
    } // namespace details

    /// deduction guide to allow the MutliRefOpt to be build without the use of sham::to_opt_ref
    template<class... Targ>
    MultiRefOpt(Targ... arg) -> MultiRefOpt<typename details::mapper<Targ>::type...>;

    /**
     * @brief A class that references multiple buffers or similar objects.
     *
     * This class serves as a means to pass multiple buffers or objects with similar accessor
     * patterns to a kernel. It provides methods to obtain read and write access to these
     * entities and to complete their event state.
     *
     * A version of this class is also available for optional references to the buffers or similar
     * objects, @see MultiRefOpt.
     */
    template<class... Targ>
    struct MultiRef {
        /// A tuple of references to the buffers.
        using storage_t = std::tuple<Targ &...>;

        /// A tuple of references to the buffers.
        storage_t storage;

        /// Constructor
        MultiRef(Targ &...arg) : storage(arg...) {}

        /// Get a tuple of pointers to the data of the buffers, for reading. Register also the
        /// depedancies in depends_list.
        auto get_read_access(sham::EventList &depends_list) {
            __shamrock_stack_entry();
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::get_read_access(__a, depends_list)...);
                },
                storage);
        }

        /// Get a tuple of pointers to the data of the buffers, for writing. Register also the
        /// depedancies in depends_list.
        auto get_write_access(sham::EventList &depends_list) {
            __shamrock_stack_entry();
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::get_write_access(__a, depends_list)...);
                },
                storage);
        }

        /// Complete the event state of the buffers.
        /// @param e The SYCL event to register in the buffers.
        void complete_event_state(sycl::event e) {
            __shamrock_stack_entry();
            std::apply(
                [&](auto &...__in) {
                    ((details::complete_event_state(__in, e)), ...);
                },
                storage);
        }
    };

    namespace details {

        /// internal implementation of typed_index_kernel_call
        template<class index_t, class RefIn, class RefOut, class Functor>
        void typed_index_kernel_call_lambda(
            sham::DeviceQueue &q,
            RefIn in,
            RefOut in_out,
            index_t n,
            Functor &&kernel_gen,
            SourceLocation &&callsite = SourceLocation{}) {

            __shamrock_stack_entry_with_callsite(callsite);

            if (n == 0) {
                shambase::throw_with_loc<std::runtime_error>("kernel call with : n == 0");
            }

            sham::EventList depends_list;

            auto acc_in     = in.get_read_access(depends_list);
            auto acc_in_out = in_out.get_write_access(depends_list);

            sycl::event e;

            // unpack the tuples of accessors
            std::apply(
                [&](auto &...__acc_in) {
                    std::apply(
                        [&](auto &...__acc_in_out) {
                            // submit the kernel generated by the functor
                            e = q.submit(depends_list, kernel_gen(n, __acc_in..., __acc_in_out...));
                        },
                        acc_in_out);
                },
                acc_in);

            in.complete_event_state(e);
            in_out.complete_event_state(e);
        }

        /// internal implementation of typed_index_kernel_call
        template<class index_t, class RefIn, class RefOut, class Functor>
        void typed_index_kernel_call(
            sham::DeviceQueue &q,
            RefIn in,
            RefOut in_out,
            index_t n,
            Functor &&func,
            SourceLocation &&callsite = SourceLocation{}) {

            __shamrock_log_callsite(callsite);

            typed_index_kernel_call_lambda(
                q,
                in,
                in_out,
                n,
                [func
                 = std::forward<Functor>(func)](u32 n, auto... __acc_in, auto... __acc_in_out) {
                    return [=](sycl::handler &cgh) {
                        cgh.parallel_for(sycl::range<1>{n}, [=](sycl::item<1> item) {
                            shambase::check_functor_signature_deduce<void>(
                                func, index_t(item.get_linear_id()), __acc_in..., __acc_in_out...);

                            func(index_t(item.get_linear_id()), __acc_in..., __acc_in_out...);
                        });
                    };
                });
        }
    } // namespace details

    /**
     * @brief Submit a kernel to a SYCL queue.
     *
     * # Automatic kernel dependency handling
     *
     * This pr introduce a kernel call function to automatically forward buffer pointers and handle
     * events, the ideal usage would be :
     * @code
     * kernel_call(queue, input buf ..., out buf ..., element count, kernel);
     * @endcode
     *
     * However, c++ does not allow multiple parameter pack so a `MultiRef` wrapper is introduced,
     * the call then looks like:
     * @code
     * kernel_call(queue, MultiRef{input buf ...}, MultiRef{out buf ...}, element count, kernel);
     * @endcode
     *
     * This allows the flexibility of forwarding more complex structures, as well as optional
     * buffers.
     *
     * ## Standard usage
     * In a normal usage it is used like so
     * @code {.cpp}
     * sham::DeviceBuffer<Tscal> &buf_P  = storage.pressure.get().get_buf_check(id);
     * sham::DeviceBuffer<Tscal> &buf_cs = storage.soundspeed.get().get_buf_check(id);
     *
     * sham::DeviceBuffer<Tscal> &buf_h = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);
     * sham::DeviceBuffer<Tscal> &buf_uint = mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf);
     *
     * sham::kernel_call(q,
     *     sham::MultiRef{buf_h, buf_uint},
     *     sham::MultiRef{buf_P, buf_cs},
     *     mpdat.total_elements,
     *     [pmass = gpart_mass, gamma = eos_config->gamma](
     *         u32 i,
     *         const Tscal *h,
     *         const Tscal *U,
     *         Tscal *P,
     *         Tscal *cs) {
     *         Tscal rho_a = rho(i);
     *         Tscal P_a   = EOS::pressure(gamma, rho_a, U[i]);
     *         Tscal cs_a  = EOS::cs_from_p(gamma, rho_a, P_a);
     *         P[i]        = P_a;
     *         cs[i]       = cs_a;
     *     });
     * @endcode
     *
     * Under the hood read and write access as well as complete_event_state will be called
     * implicitly thanks to the template resolution.
     *
     * ## Complex accessors
     * Since `sham::kernel_call` simply call get_read_access, get_write_access,
     * complete_event_state. We can pass a complex struct instead of a `DeviceBuffer` as long as it
     * defines similar accessor functions.
     *
     * Example :
     * @code {.cpp}
     * sham::DeviceBuffer<Tscal> &buf_P  = storage.pressure.get().get_buf_check(id);
     * sham::DeviceBuffer<Tscal> &buf_cs = storage.soundspeed.get().get_buf_check(id);
     *
     * sham::DeviceBuffer<Tscal> &buf_h = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);
     * sham::DeviceBuffer<Tscal> &buf_uint = mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf);
     *
     * struct RhoGetter {
     *     sham::DeviceBuffer<Tscal> &buf_h;
     *     Tscal pmass;
     *     Tscal hfact;
     *
     *     struct accessed {
     *         const Tscal *h;
     *         Tscal pmass;
     *         Tscal hfact;
     *
     *         Tscal operator()(u32 i) const {
     *             using namespace shamrock::sph;
     *             return rho_h(pmass, h[i], hfact);
     *         }
     *     };
     *
     *     accessed get_read_access(sham::EventList &depends_list) {
     *         auto h = buf_h.get_read_access(depends_list);
     *         return accessed{h, pmass, hfact};
     *     }
     *
     *     void complete_event_state(sycl::event e) { buf_h.complete_event_state(e);}
     * };
     *
     * RhoGetter rho_getter{buf_h, gpart_mass, Kernel::hfactd};
     *
     * sham::kernel_call(q,
     *     sham::MultiRef{rho_getter, buf_uint},
     *     sham::MultiRef{buf_P, buf_cs},
     *     mpdat.total_elements,
     *     [gamma = eos_config->gamma](
     *         u32 i,
     *         const typename RhoGetter::accessed rho,
     *         const Tscal *U,
     *         Tscal *P,
     *         Tscal *cs) {
     *         Tscal rho_a = rho(i);
     *         Tscal P_a   = EOS::pressure(gamma, rho_a, U[i]);
     *         Tscal cs_a  = EOS::cs_from_p(gamma, rho_a, P_a);
     *         P[i]        = P_a;
     *         cs[i]       = cs_a;
     *     });
     * @endcode
     *
     * ## Optional arguments
     * Another type of `MultiRef` called `MultiRefOpt` can be introduced to pass optional buffers to
     * have buffer specialization thanks to dead argument elimination.
     *
     * It can be used as follows:
     * @code {.cpp}
     * sham::DeviceBuffer<Tscal> &buf_P  = storage.pressure.get().get_buf_check(id);
     * sham::DeviceBuffer<Tscal> &buf_cs = storage.soundspeed.get().get_buf_check(id);
     *
     * sham::DeviceBuffer<Tscal> &buf_h = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);
     * sham::DeviceBuffer<Tscal> &buf_uint = mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf);
     *
     * auto get_eps = [&]() {
     *     if constexpr (is_monofluid) {
     *         sham::DeviceBuffer<Tscal> &buf_epsilon
     *             = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);
     *         return to_opt_ref(buf_epsilon);
     *     } else {
     *         return empty_buf_ref<Tscal>();
     *     }
     * };
     *
     * sham::kernel_call(q,
     *     sham::MultiRefOpt{buf_h, buf_uint, get_eps()},
     *     sham::MultiRef{buf_P, buf_cs},
     *     mpdat.total_elements,
     *     [pmass = gpart_mass, gamma = eos_config->gamma](
     *         u32 i,
     *         const Tscal *h,
     *         const Tscal *U,
     *         const Tscal *epsilon, // set to nullptr if not is_monofluid
     *         Tscal *P,
     *         Tscal *cs) {
     *         auto rho = [&]() {
     *             using namespace shamrock::sph;
     *             if constexpr (is_monofluid) {
     *                 return (1 - epsilon[i]) * rho_h(pmass, h[i], Kernel::hfactd);
     *             } else {
     *                 return rho_h(pmass, h[i], Kernel::hfactd);
     *             }
     *         };
     *
     *         Tscal rho_a = rho();
     *         Tscal P_a   = EOS::pressure(gamma, rho_a, U[i]);
     *         Tscal cs_a  = EOS::cs_from_p(gamma, rho_a, P_a);
     *         P[i]        = P_a;
     *         cs[i]       = cs_a;
     *     });
     * @endcode
     *
     * @param q The SYCL queue to submit the kernel to.
     * @param in The input buffer or MultiRef or MultiRefOpt.
     * @param in_out The input/output buffer or MultiRef or MultiRefOpt.
     * @param n The number of thread to launch.
     * @param func The functor to call for each thread launched.
     */
    template<class RefIn, class RefOut, class Functor>
    void kernel_call(
        sham::DeviceQueue &q,
        RefIn in,
        RefOut in_out,
        u32 n,
        Functor &&func,
        SourceLocation &&callsite = SourceLocation{}) {

        __shamrock_log_callsite(callsite);

        details::typed_index_kernel_call<u32, RefIn, RefOut>(
            q, in, in_out, n, std::forward<Functor>(func));
    }

    /// u64 indexed variant of kernel_call
    template<class RefIn, class RefOut, class Functor>
    void kernel_call_u64(
        sham::DeviceQueue &q,
        RefIn in,
        RefOut in_out,
        u64 n,
        Functor &&func,
        SourceLocation &&callsite = SourceLocation{}) {

        __shamrock_log_callsite(callsite);

        details::typed_index_kernel_call<u64, RefIn, RefOut>(
            q, in, in_out, n, std::forward<Functor>(func));
    }

    // version where one supplies a kernel generator in the form of [&](sycl::handler &cgh) { ... }
    template<class RefIn, class RefOut, class Functor>
    void kernel_call_hndl(
        sham::DeviceQueue &q,
        RefIn in,
        RefOut in_out,
        u32 n,
        Functor &&kernel_gen,
        SourceLocation &&callsite = SourceLocation{}) {

        __shamrock_log_callsite(callsite);

        details::typed_index_kernel_call_lambda<u32, RefIn, RefOut>(
            q, in, in_out, n, std::forward<Functor>(kernel_gen));
    }

    /// u64 indexed variant of kernel_call_hndl
    template<class RefIn, class RefOut, class Functor>
    void kernel_call_hndl_u64(
        sham::DeviceQueue &q,
        RefIn in,
        RefOut in_out,
        u64 n,
        Functor &&kernel_gen,
        SourceLocation &&callsite = SourceLocation{}) {

        __shamrock_log_callsite(callsite);

        details::typed_index_kernel_call_lambda<u64, RefIn, RefOut>(
            q, in, in_out, n, std::forward<Functor>(kernel_gen));
    }

} // namespace sham
