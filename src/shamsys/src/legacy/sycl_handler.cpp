// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sycl_handler.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/string.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <unordered_map>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

auto check_queue_is_valid(sycl::queue &q) {

    auto test_kernel = [](sycl::queue &q) {
        sycl::buffer<u32> b{10};

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor acc{b, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{1}, [=](sycl::item<1> i) {
                acc[i] = i.get_linear_id();
            });
        });

        q.wait();
    };

    std::exception_ptr eptr;
    try {
        test_kernel(q);
        logger::info_ln(
            "SYCL Hanlder",
            "selected queue :",
            q.get_device().get_info<sycl::info::device::name>(),
            " working !");
    } catch (...) {
        eptr = std::current_exception(); // capture
    }

    if (eptr) {
        logger::err_ln(
            "SYCL Hanlder",
            "selected queue :",
            q.get_device().get_info<sycl::info::device::name>(),
            "does not function properly");
        std::rethrow_exception(eptr);
    }
}

auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const &e) {
            printf("Caught synchronous SYCL exception: %s\n", e.what());
        }
    }
};

std::string getDeviceTypeName(const sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    switch (DeviceType) {
    case sycl::info::device_type::cpu        : return "CPU";
    case sycl::info::device_type::gpu        : return "GPU";
    case sycl::info::device_type::host       : return "HOST";
    case sycl::info::device_type::accelerator: return "ACCELERATOR";
    default                                  : return "UNKNOWN";
    }
}

void print_device_info(const sycl::device &Device) {
    std::cout << "   - " << Device.get_info<sycl::info::device::name>() << " "
              << shambase::readable_sizeof(Device.get_info<sycl::info::device::global_mem_size>())
              << "\n";
}

namespace sycl_handler {
    bool already_on = false;

    std::unique_ptr<sycl::queue> compute_queue;
    std::unique_ptr<sycl::queue> alt_queue;

    sycl::queue &get_compute_queue() {
        if (!already_on) {
            throw ShamrockSyclException("sycl handler is not initialized");
        }
        return *compute_queue;
    }

    sycl::queue &get_alt_queue() {
        if (!already_on) {
            throw ShamrockSyclException("sycl handler is not initialized");
        }
        return *alt_queue;
    }

    void init() {

        if (already_on) {
            throw ShamrockSyclException("Sycl Handler is already on");
        }

        logger::raw_ln(
            shambase::term_colors::col8b_cyan() + " >>> init SYCL instances <<< "
            + shambase::term_colors::reset());

        if (opts::has_option("--sycl-cfg")) {
            std::string sycl_cfg = std::string(opts::get_option("--sycl-cfg"));

            logger::normal_ln("SYCL Handler", "chosen sycl config :", sycl_cfg);

            size_t split_alt_comp = 0;
            split_alt_comp        = sycl_cfg.find(":");

            if (split_alt_comp == std::string::npos) {
                logger::err_ln("SYCL Handler", "sycl-cfg layout should be x:x");
                throw ShamrockSyclException("sycl-cfg layout should be x:x");
            }

            std::string alt_cfg  = sycl_cfg.substr(0, split_alt_comp);
            std::string comp_cfg = sycl_cfg.substr(split_alt_comp + 1, sycl_cfg.length());

            i32 ialt, icomp;
            try {
                try {
                    ialt = std::stoi(alt_cfg);
                } catch (const std::invalid_argument &a) {
                    logger::err_ln("SYCL Handler", "alt config is not an int");
                    throw ShamrockSyclException("alt config is not an int");
                }
            } catch (const std::out_of_range &a) {
                logger::err_ln("SYCL Handler", "alt config is to big for an integer");
                throw ShamrockSyclException("alt config is to big for an integer");
            }

            try {
                try {
                    icomp = std::stoi(comp_cfg);
                } catch (const std::invalid_argument &a) {
                    logger::err_ln("SYCL Handler", "compute config is not an int");
                    throw ShamrockSyclException("compute config is not an int");
                }
            } catch (const std::out_of_range &a) {
                logger::err_ln("SYCL Handler", "compute config is to big for an integer");
                throw ShamrockSyclException("compute config is to big for an integer");
            }

            const auto &Platforms = sycl::platform::get_platforms();

            logger::raw_ln(
                "--------------------------------------------------------------------------------");
            logger::raw_ln(
                "| sel | id |        Device name            |       Platform name      |  Type  |");
            logger::raw_ln(
                "--------------------------------------------------------------------------------");

            u32 key_global = 0;
            for (const auto &Platform : Platforms) {
                const auto &Devices = Platform.get_devices();

                auto PlatformName = Platform.get_info<sycl::info::platform::name>();
                for (const auto &Device : Devices) {
                    auto DeviceName = Device.get_info<sycl::info::device::name>();

                    auto selected_k = [&](i32 k) {
                        std::string ret = "";

                        if (k == ialt) {
                            ret += "a";
                        }

                        if (k == icomp) {
                            ret += "c";
                        }

                        return ret;
                    };

                    std::string selected = selected_k(key_global);

                    std::string devname  = shambase::trunc_str(DeviceName, 29);
                    std::string platname = shambase::trunc_str(PlatformName, 24);
                    std::string devtype  = shambase::trunc_str(getDeviceTypeName(Device), 6);

                    logger::raw_ln(
                        shambase::format_printf(
                            "| %-3s | %02d | %-29s | %-24s | %-6s |",
                            selected.c_str(),
                            key_global,
                            devname.c_str(),
                            platname.c_str(),
                            devtype.c_str()));

                    if (key_global == ialt) {
                        alt_queue = std::make_unique<sycl::queue>(Device, exception_handler);
                    }

                    if (key_global == icomp) {
                        compute_queue = std::make_unique<sycl::queue>(Device, exception_handler);
                    }

                    key_global++;
                }
            }

            logger::raw_ln(
                "--------------------------------------------------------------------------------");

            key_global = 0;
            for (const auto &Platform : Platforms) {
                auto PlatformName = Platform.get_info<sycl::info::platform::name>();
                for (const auto &Device : Platform.get_devices()) {
                    auto DeviceName = Device.get_info<sycl::info::device::name>();

                    if (key_global == ialt) {
                        logger::info_ln(
                            "SYCL Handler",
                            "init alt queue  : ",
                            "|",
                            DeviceName,
                            "|",
                            PlatformName,
                            "|",
                            getDeviceTypeName(Device),
                            "|");
                        alt_queue = std::make_unique<sycl::queue>(Device, exception_handler);
                    }

                    if (key_global == icomp) {
                        logger::info_ln(
                            "SYCL Handler",
                            "init comp queue : ",
                            "|",
                            DeviceName,
                            "|",
                            PlatformName,
                            "|",
                            getDeviceTypeName(Device),
                            "|");
                        compute_queue = std::make_unique<sycl::queue>(Device, exception_handler);
                    }

                    key_global++;
                }
            }

            check_queue_is_valid(*compute_queue);
            check_queue_is_valid(*alt_queue);

            logger::info_ln("SYCL Handler", "init done");

            logger::info_ln("SYCL Handler", "creating MPI type for interop");
            create_sycl_mpi_types();

            logger::info_ln("SYCL Handler", "MPI type for interop created");

        } else {

            logger::err_ln("SYCL Handler", "Please specify a sycl configuration (--sycl-cfg x:x)");
            // std::cout << "[SYCL Handler] Please specify a sycl configuration (--sycl-cfg x:x)" <<
            // std::endl;
            throw ShamrockSyclException("Sycl Handler need configuration (--sycl-cfg x:x)");
        }

        already_on = true;
    }
} // namespace sycl_handler
