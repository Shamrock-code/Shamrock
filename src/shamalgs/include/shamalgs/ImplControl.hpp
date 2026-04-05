// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ImplControl.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceScheduler.hpp"
#include "shamcomm/logs.hpp"
#include <string>

namespace shamalgs::primitives {
    class ImplControl {
        public:
        virtual ~ImplControl() = default;

        // public API
        inline std::string get_alg_name() const { return impl_get_alg_name(); }

        inline bool was_configured(const sham::DeviceScheduler_ptr &dev_sched) const {
            return impl_was_configured(dev_sched);
        }

        inline std::string get_config(const sham::DeviceScheduler_ptr &dev_sched) {
            if (!impl_was_configured(dev_sched)) {
                set_config(dev_sched, get_default_config(dev_sched));
            }
            return impl_get_config(dev_sched);
        }

        inline void set_config(const sham::DeviceScheduler_ptr &dev_sched, const std::string &cfg) {
            logger::info_ln(
                "Algs",
                shambase::format(
                    "switching config for alg {} to cfg={}", impl_get_alg_name(), cfg));
            impl_set_config(dev_sched, cfg);
        }

        inline std::string get_default_config(const sham::DeviceScheduler_ptr &dev_sched) {
            if (auto cfg = impl_autotune(dev_sched)) {
                return *cfg;
            } else {
                return impl_get_default_config(dev_sched);
            }
        }

        inline std::vector<std::string> get_avail_configs(
            const sham::DeviceScheduler_ptr &dev_sched) {
            return impl_get_avail_configs(dev_sched);
        };

        protected:
        // required overrides
        virtual std::string impl_get_alg_name() const                                           = 0;
        virtual bool impl_was_configured(const sham::DeviceScheduler_ptr &) const               = 0;
        virtual std::string impl_get_config(const sham::DeviceScheduler_ptr &) const            = 0;
        virtual std::string impl_get_default_config(const sham::DeviceScheduler_ptr &) const    = 0;
        virtual void impl_set_config(const sham::DeviceScheduler_ptr &, const std::string &cfg) = 0;
        virtual std::vector<std::string> impl_get_avail_configs(const sham::DeviceScheduler_ptr &)
            = 0;

        // optional override
        inline virtual std::optional<std::string> impl_autotune(const sham::DeviceScheduler_ptr &) {
            logger::info_ln("Algs", "no autotuning registered for", impl_get_alg_name());
            return std::nullopt;
        }
    };
} // namespace shamalgs::primitives
