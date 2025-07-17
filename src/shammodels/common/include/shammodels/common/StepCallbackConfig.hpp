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
 * @file StepCallbackConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/logs.hpp"
#include "nlohmann/json.hpp"
#include "shambackends/vec.hpp"
#include <stdexcept>
#include <variant>
#include <vector>

namespace shammodels {

    template<class Tscal>
    struct StepCallback {
        std::string name;

        /// Frequency (time) at which the callback is called (-1 = never)
        Tscal callback_time_freq = -1;

        /// Frequency (step) at which the callback is called (-1 = never)
        i64 callback_step_freq = -1;
    };

    template<class Tscal>
    struct StepCallbackState {
        std::string name;

        /// Frequency (time) at which the callback is called (-1 = never)
        Tscal next_call_time = -1;

        /// Frequency (step) at which the callback is called (-1 = never)
        i64 next_step = -1;

        /// Number of times the callback was called
        u64 call_count = 0;
    };

    /// (icall, istep, t)
    template<class Tscal>
    using step_callback_func_t = std::function<void(u64, u64, Tscal)>;

    template<class Tscal>
    struct StepCallbackConfig {

        std::unordered_map<std::string, StepCallback<Tscal>> step_callback_config;
        std::unordered_map<std::string, StepCallbackState<Tscal>> step_callback_state;
        std::unordered_map<std::string, step_callback_func_t<Tscal>> step_callbacks;

        inline void limit_dt(Tscal t, Tscal &dt) {
            for (const auto &[key, func] : step_callback_state) {
                if (func.next_call_time == -1) {
                    continue;
                }
                auto delta = func.next_call_time - t;

                if (delta >= 0) {
                    dt = std::min(dt, delta);
                } else {
                    shambase::throw_with_loc<std::runtime_error>(shambase::format(
                        "step callback next time is in the past"
                        " (delta < 0)\n"
                        "    t = {}\n"
                        "    delta = {}\n"
                        "    func.next_call_time = {}\n"
                        "    t = {}",
                        t,
                        delta,
                        func.next_call_time,
                        t));
                }
            }
        }

        inline void handle_callbacks(Tscal t, u64 istep) {

            struct call {
                u64 icall;
                u64 istep;
                Tscal t;
                step_callback_func_t<Tscal> &func;

                inline void do_call() { func(icall, istep, t); }
            };

            std::vector<call> to_call;

            for (const auto &[key, cfg] : step_callback_config) {

                step_callback_func_t<Tscal> &func = step_callbacks.at(key);
                StepCallbackState<Tscal> &state   = step_callback_state.at(key);

                bool do_call_time = (state.next_call_time <= t && state.next_call_time != -1);
                bool do_call_step = (state.next_step <= istep && state.next_step != -1);

                if (do_call_time || do_call_step) {

                    auto _icall = state.call_count;
                    auto _istep = istep;
                    auto _t     = t;

                    if (do_call_time) {
                        state.next_call_time += cfg.callback_time_freq;
                    }
                    if (do_call_step) {
                        state.next_step += cfg.callback_step_freq;
                    }

                    state.call_count++;

                    to_call.push_back(call{_icall, _istep, _t, func});
                }
            }

            for (auto &call : to_call) {
                call.do_call();
            }
        }

        inline void register_callback(
            std::string name,
            Tscal callback_time_freq   = -1,
            i64 callback_step_freq     = -1,
            Tscal until_next_call_time = -1,
            i64 until_next_step        = -1) {

            if (until_next_call_time == -1) {
                until_next_call_time = 0;
            }
            if (until_next_step == -1) {
                until_next_step = 0;
            }

            if (step_callback_config.find(name) == step_callback_config.end()) {
                step_callback_config[name] = {name, callback_time_freq, callback_step_freq};
                step_callback_state[name]  = {name, until_next_call_time, until_next_step};
            } else {
                shambase::throw_with_loc<std::runtime_error>(
                    "callback config already contains key " + name);
            }
        }

        inline void attach_callback(std::string name, step_callback_func_t<Tscal> callback) {

            if (step_callbacks.find(name) != step_callbacks.end()) {
                shambase::throw_with_loc<std::runtime_error>(
                    "callbacks already contains key " + name);
            }

            if (step_callback_config.find(name) == step_callback_config.end()) {
                shambase::throw_with_loc<std::runtime_error>(
                    "you want to attach a callback for key " + name + " but it was not registered");
            }

            step_callbacks[name] = callback;
        }

        inline void validate_config() {

            for (const auto &[key, _] : step_callback_config) {
                if (step_callback_config.find(key) == step_callback_config.end()) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "callback config does not contain key " + key);
                }
                if (step_callback_state.find(key) == step_callback_state.end()) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "callback state does not contain key " + key);
                }
            }
        }
    };

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, StepCallback<Tscal> &p) {
        j.at("name").get_to(p.name);
        j.at("callback_time_freq").get_to(p.callback_time_freq);
        j.at("callback_step_freq").get_to(p.callback_step_freq);
    }

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const StepCallback<Tvec> &p) {
        j = nlohmann::json{
            {"name", p.name},
            {"callback_time_freq", p.callback_time_freq},
            {"callback_step_freq", p.callback_step_freq},
        };
    }

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, StepCallbackState<Tscal> &p) {
        j.at("name").get_to(p.name);
        j.at("next_call_time").get_to(p.next_call_time);
        j.at("next_step").get_to(p.next_step);
        j.at("call_count").get_to(p.call_count);
    }

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const StepCallbackState<Tvec> &p) {
        j = nlohmann::json{
            {"name", p.name},
            {"next_call_time", p.next_call_time},
            {"next_step", p.next_step},
            {"call_count", p.call_count},
        };
    }

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, StepCallbackConfig<Tscal> &p) {
        j.at("step_callback_config").get_to(p.step_callback_config);
        j.at("step_callback_state").get_to(p.step_callback_state);
    }

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const StepCallbackConfig<Tvec> &p) {
        j = nlohmann::json{
            {"step_callback_config", p.step_callback_config},
            {"step_callback_state", p.step_callback_state},
        };
    }

} // namespace shammodels
