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
 * @file CommunicationBufferImpl.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include <stdexcept>
#include <variant>

namespace shamcomm {

    enum CommunicationProtocol {
        /**
         * @brief copy data to the host and then perform the call
         */
        CopyToHost,

        /**
         * @brief copy data straight from the GPU
         */
        DirectGPU,

    };

    inline CommunicationProtocol get_protocol(sham::Device &device) {
        if (device.mpi_prop.is_mpi_direct_capable) {
            return DirectGPU;
        } else {
            return CopyToHost;
        }
    }

    namespace details {

        template<CommunicationProtocol comm_mode>
        class CommunicationBuffer;

        template<>
        class CommunicationBuffer<CopyToHost> {

            std::shared_ptr<sham::DeviceScheduler> dev_sched;

            sham::DeviceBuffer<u8, sham::host> usm_buf;

            public:
            inline CommunicationBuffer(
                u64 bytelen, std::shared_ptr<sham::DeviceScheduler> dev_sched)
                : dev_sched(dev_sched), usm_buf(bytelen, dev_sched) {}

            inline CommunicationBuffer(
                sycl::buffer<u8> &obj_ref, std::shared_ptr<sham::DeviceScheduler> dev_sched)
                : dev_sched(dev_sched), usm_buf(obj_ref, dev_sched) {}

            inline CommunicationBuffer(
                sycl::buffer<u8> &&moved_obj, std::shared_ptr<sham::DeviceScheduler> dev_sched)
                : dev_sched(dev_sched), usm_buf(moved_obj, dev_sched) {}

            inline CommunicationBuffer(
                sham::DeviceBuffer<u8, sham::host> &&moved_obj,
                std::shared_ptr<sham::DeviceScheduler> dev_sched)
                : dev_sched(dev_sched),
                  usm_buf(std::forward<sham::DeviceBuffer<u8, sham::host>>(moved_obj)) {}

            inline std::unique_ptr<CommunicationBuffer> duplicate_to_ptr() {
                std::unique_ptr<CommunicationBuffer> ret
                    = std::make_unique<CommunicationBuffer>(usm_buf.copy(), dev_sched);
                return ret;
            }

            inline u64 get_size() { return usm_buf.get_size(); }

            inline sycl::buffer<u8> copy_back() { return usm_buf.copy_to_sycl_buffer(); }
            static sycl::buffer<u8> convert(CommunicationBuffer &&buf) { return buf.copy_back(); }

            inline sham::DeviceBuffer<u8> copy_back_usm() {
                return usm_buf.copy_to<sham::device>();
            }
            static sham::DeviceBuffer<u8> convert_usm(CommunicationBuffer &&buf) {
                return buf.usm_buf.copy_to<sham::device>();
            }

            u8 *get_ptr() {
                sham::EventList depends_list;
                u8 *ptr = usm_buf.get_write_access(depends_list);
                depends_list.wait_and_throw();
                usm_buf.complete_event_state(sycl::event{});

                return ptr;
            }
        };

        template<>
        class CommunicationBuffer<DirectGPU> {

            std::shared_ptr<sham::DeviceScheduler> dev_sched;

            sham::DeviceBuffer<u8, sham::device> usm_buf;

            public:
            inline CommunicationBuffer(
                u64 bytelen, std::shared_ptr<sham::DeviceScheduler> dev_sched)
                : dev_sched(dev_sched), usm_buf(bytelen, dev_sched) {}

            inline CommunicationBuffer(
                sycl::buffer<u8> &obj_ref, std::shared_ptr<sham::DeviceScheduler> dev_sched)
                : dev_sched(dev_sched), usm_buf(obj_ref, dev_sched) {}

            inline CommunicationBuffer(
                sycl::buffer<u8> &&moved_obj, std::shared_ptr<sham::DeviceScheduler> dev_sched)
                : dev_sched(dev_sched), usm_buf(moved_obj, dev_sched) {}

            inline CommunicationBuffer(
                sham::DeviceBuffer<u8, sham::device> &&moved_obj,
                std::shared_ptr<sham::DeviceScheduler> dev_sched)
                : dev_sched(dev_sched),
                  usm_buf(std::forward<sham::DeviceBuffer<u8, sham::device>>(moved_obj)) {}

            inline std::unique_ptr<CommunicationBuffer> duplicate_to_ptr() {
                std::unique_ptr<CommunicationBuffer> ret
                    = std::make_unique<CommunicationBuffer>(usm_buf.copy(), dev_sched);
                return ret;
            }

            inline sycl::buffer<u8> copy_back() { return usm_buf.copy_to_sycl_buffer(); }
            inline sham::DeviceBuffer<u8> copy_back_usm() {
                return usm_buf.copy_to<sham::device>();
            }

            inline u64 get_size() { return usm_buf.get_size(); }

            static sycl::buffer<u8> convert(CommunicationBuffer &&buf) { return buf.copy_back(); }
            static sham::DeviceBuffer<u8> convert_usm(CommunicationBuffer &&buf) {
                return std::move(buf.usm_buf);
            }

            u8 *get_ptr() {
                sham::EventList depends_list;
                u8 *ptr = usm_buf.get_write_access(depends_list);
                depends_list.wait_and_throw();
                usm_buf.complete_event_state(sycl::event{});

                return ptr;
            }
        };
    } // namespace details

} // namespace shamcomm
