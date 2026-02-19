// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file device_select.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamcomm/env_variables.hpp"
#include "shamsys/device_select.hpp"
#include "shamsys/for_each_device.hpp"

shamsys::DeviceSelectRet_t init_queues_auto(std::string search_key) {

    StackEntry stack_loc{false};

    shamsys::DeviceSelectRet_t ret;

    std::optional<u32> local_id = shamcomm::get_local_rank();

    if (local_id) {

        u32 valid_dev_cnt = 0;

        shamsys::for_each_device([&](u32 key_global,
                                     const sycl::platform &plat,
                                     const sycl::device &dev) {
            if (shambase::contain_substr(plat.get_info<sycl::info::platform::name>(), search_key)) {
                valid_dev_cnt++;
            }
        });

        u32 valid_dev_id = 0;

        shamsys::for_each_device([&](u32 key_global,
                                     const sycl::platform &plat,
                                     const sycl::device &dev) {
            if (shambase::contain_substr(plat.get_info<sycl::info::platform::name>(), search_key)) {

                if ((*local_id) % valid_dev_cnt == valid_dev_id) {
                    shamlog_debug_sycl_ln(
                        "Sys",
                        "create queue :\n",
                        "Local ID :",
                        *local_id,
                        "\n Queue id :",
                        key_global);

                    auto PlatformName = plat.get_info<sycl::info::platform::name>();
                    auto DeviceName   = dev.get_info<sycl::info::device::name>();
                    shamlog_debug_sycl_ln(
                        "NodeInstance",
                        "init alt queue  : ",
                        "|",
                        DeviceName,
                        "|",
                        PlatformName,
                        "|",
                        shambase::getDevice_type(dev),
                        "|");

                    ret.device_alt = std::make_shared<sham::Device>(
                        sham::sycl_dev_to_sham_dev(key_global, dev));

                    shamlog_debug_sycl_ln(
                        "NodeInstance",
                        "init comp queue : ",
                        "|",
                        DeviceName,
                        "|",
                        PlatformName,
                        "|",
                        shambase::getDevice_type(dev),
                        "|");
                    ret.device_compute = std::make_shared<sham::Device>(
                        sham::sycl_dev_to_sham_dev(key_global, dev));
                }

                valid_dev_id++;
            }
        });

    } else {
        logger::err_ln("Sys", "cannot query local rank cannot use autodetect");
        throw shambase::make_except_with_loc<std::runtime_error>(
            "cannot query local rank cannot use autodetect");
    }

    return ret;
}

shamsys::DeviceSelectRet_t init_queues(u32 alt_id, u32 compute_id) {

    StackEntry stack_loc{false};

    shamsys::DeviceSelectRet_t ret;

    u32 cnt_dev = shamsys::for_each_device(
        [&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {});

    if (alt_id >= cnt_dev) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "the alt queue id is larger than the number of queue");
    }

    if (compute_id >= cnt_dev) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "the compute queue id is larger than the number of queue");
    }

    shamsys::for_each_device(
        [&](u32 key_global, const sycl::platform &plat, const sycl::device &dev) {
            auto PlatformName = plat.get_info<sycl::info::platform::name>();
            auto DeviceName   = dev.get_info<sycl::info::device::name>();

            if (key_global == alt_id) {
                shamlog_debug_sycl_ln(
                    "NodeInstance",
                    "init alt queue  : ",
                    "|",
                    DeviceName,
                    "|",
                    PlatformName,
                    "|",
                    shambase::getDevice_type(dev),
                    "|");
                ret.device_alt
                    = std::make_shared<sham::Device>(sham::sycl_dev_to_sham_dev(key_global, dev));
            }

            if (key_global == compute_id) {
                shamlog_debug_sycl_ln(
                    "NodeInstance",
                    "init comp queue : ",
                    "|",
                    DeviceName,
                    "|",
                    PlatformName,
                    "|",
                    shambase::getDevice_type(dev),
                    "|");
                ret.device_compute
                    = std::make_shared<sham::Device>(sham::sycl_dev_to_sham_dev(key_global, dev));
            }
        });

    return ret;
}
namespace shamsys {

    /**
     * @brief Select the devices for the queues
     *
     * If the config string starts with "auto:", then the function
     * init_queues_auto is called with the remaining string as argument.
     * Otherwise, the config string is split at the first colon, and the
     * integers on the left and right are used as arguments to the function
     * init_queues.
     *
     * @param sycl_cfg the config string
     * @return a DeviceSelectRet_t containing the selected devices
     */
    DeviceSelectRet_t select_devices(std::string sycl_cfg) {

        if (shambase::contain_substr(sycl_cfg, "auto:")) {

            std::string search = sycl_cfg.substr(5);
            return init_queues_auto(search);

        } else {

            size_t split_alt_comp = 0;
            split_alt_comp        = sycl_cfg.find(":");

            if (split_alt_comp == std::string::npos) {
                logger::err_ln("NodeInstance", "sycl-cfg layout should be x:x");
                shambase::throw_with_loc<std::runtime_error>("sycl-cfg layout should be x:x");
            }

            std::string alt_cfg  = sycl_cfg.substr(0, split_alt_comp);
            std::string comp_cfg = sycl_cfg.substr(split_alt_comp + 1, sycl_cfg.length());

            u32 ialt, icomp;
            try {
                try {
                    ialt = std::stoi(alt_cfg);
                } catch (const std::invalid_argument &a) {
                    logger::err_ln("NodeInstance", "alt config is not an int");
                    shambase::throw_with_loc<std::runtime_error>("alt config is not an int");
                }
            } catch (const std::out_of_range &a) {
                logger::err_ln("NodeInstance", "alt config is to big for an integer");
                shambase::throw_with_loc<std::runtime_error>("alt config is to big for an integer");
            }

            try {
                try {
                    icomp = std::stoi(comp_cfg);
                } catch (const std::invalid_argument &a) {
                    logger::err_ln("NodeInstance", "compute config is not an int");
                    shambase::throw_with_loc<std::runtime_error>("compute config is not an int");
                }
            } catch (const std::out_of_range &a) {
                logger::err_ln("NodeInstance", "compute config is to big for an integer");
                shambase::throw_with_loc<std::runtime_error>(
                    "compute config is to big for an integer");
            }

            return init_queues(ialt, icomp);
        }
    }

} // namespace shamsys
