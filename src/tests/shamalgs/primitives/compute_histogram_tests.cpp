// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shambase/type_name_info.hpp"
#include "shamalgs/primitives/compute_histogram.hpp"
#include "shamalgs/primitives/mock_value.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <utility>
#include <vector>

template<class T>
inline bool compare(const std::vector<T> &v1, const std::vector<T> &v2, T tol) {

    if (v1.size() != v2.size()) {
        return false;
    }

    bool result = true;
    for (size_t i = 0; i < v1.size(); i++) {
        if (sham::abs(v1[i] - v2[i]) > tol) {
            // std::cout << i << " " << f64(v1[i]) << " " << f64(v2[i]) << " " << f64(v1[i] - v2[i])
            //          << std::endl;
            // logger::raw_ln(i, f64(v1[i]), f64(v2[i]), f64(v1[i] - v2[i]));
            result = false;
        }
    }

    return result;
}

template<class Tscal>
inline void basic_histogram(const std::vector<std::string> &impl_list) {

    logger::raw_ln("testing basic_histogram Tscal =", shambase::get_type_name<Tscal>());

    std::vector<Tscal> positions{};

    size_t N      = 1e5;
    size_t Narray = 2048;

    std::mt19937_64 eng(0x42);
    for (size_t i = 0; i < N; i++) {
        positions.push_back(shamalgs::primitives::mock_value<Tscal>(eng, 0, 1));
    }

    std::vector<Tscal> arr_pos_inf{};
    std::vector<Tscal> arr_pos_sup{};
    for (size_t i = 0; i < Narray; i++) {
        arr_pos_inf.push_back(Tscal(i) / Tscal(Narray));
        arr_pos_sup.push_back(Tscal(i + 1) / Tscal(Narray));
    }

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceBuffer<Tscal> bin_inf(Narray, dev_sched);
    sham::DeviceBuffer<Tscal> bin_sup(Narray, dev_sched);
    sham::DeviceBuffer<Tscal> pos(N, dev_sched);

    bin_inf.copy_from_stdvec(arr_pos_inf);
    bin_sup.copy_from_stdvec(arr_pos_sup);
    pos.copy_from_stdvec(positions);

    bin_inf.synchronize();
    bin_sup.synchronize();
    pos.synchronize();

    std::vector<Tscal> ref_result{};

    for (auto &cfg : impl_list) {
        using namespace shamalgs::primitives::impl;
        compute_histogram_impl_control.set_config(dev_sched, cfg);

        shambase::Timer timer;
        timer.start();
        sham::DeviceBuffer<Tscal> ret = shamalgs::primitives::compute_histogram<Tscal>(
            dev_sched,
            bin_inf,
            bin_sup,
            N,
            [](Tscal edge_inf, Tscal edge_sup, Tscal pos, bool &has_value) {
                has_value = edge_inf <= pos && pos < edge_sup;
                return (has_value) ? 1 : 0;
            },
            pos);
        ret.synchronize();
        timer.end();

        logger::raw_ln("impl =", cfg, "time =", timer.get_time_str());

        if (cfg == "reference") {
            ref_result = ret.copy_to_stdvec();
        } else {
            REQUIRE(compare<Tscal>(ref_result, ret.copy_to_stdvec(), 1e-12));
        }
    }
}

template<class Tscal>
inline void basic_histogram_size(const std::vector<std::string> &impl_list) {

    logger::raw_ln("testing basic_histogram_size Tscal =", shambase::get_type_name<Tscal>());

    std::vector<Tscal> positions{};
    std::vector<Tscal> sizes{};

    size_t N      = 1e5;
    size_t Narray = 2048;

    std::mt19937_64 eng(0x42);
    for (size_t i = 0; i < N; i++) {
        positions.push_back(shamalgs::primitives::mock_value<Tscal>(eng, 0, 1));
        sizes.push_back(0.05);
    }

    std::vector<Tscal> arr_pos_inf{};
    std::vector<Tscal> arr_pos_sup{};
    for (size_t i = 0; i < Narray; i++) {
        arr_pos_inf.push_back(Tscal(i) / Tscal(Narray));
        arr_pos_sup.push_back(Tscal(i + 1) / Tscal(Narray));
    }

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceBuffer<Tscal> bin_inf(Narray, dev_sched);
    sham::DeviceBuffer<Tscal> bin_sup(Narray, dev_sched);
    sham::DeviceBuffer<Tscal> pos(N, dev_sched);
    sham::DeviceBuffer<Tscal> szs(N, dev_sched);

    bin_inf.copy_from_stdvec(arr_pos_inf);
    bin_sup.copy_from_stdvec(arr_pos_sup);
    pos.copy_from_stdvec(positions);
    szs.copy_from_stdvec(sizes);

    bin_inf.synchronize();
    bin_sup.synchronize();
    pos.synchronize();
    szs.synchronize();

    std::vector<Tscal> ref_result{};

    for (auto &cfg : impl_list) {
        using namespace shamalgs::primitives::impl;
        compute_histogram_impl_control.set_config(dev_sched, cfg);

        shambase::Timer timer;
        timer.start();
        sham::DeviceBuffer<Tscal> ret = shamalgs::primitives::compute_histogram<Tscal>(
            dev_sched,
            bin_inf,
            bin_sup,
            N,
            [](Tscal edge_inf, Tscal edge_sup, Tscal pos, Tscal sizes, bool &has_value) {
                has_value = edge_inf - sizes <= pos && pos < edge_sup + sizes;
                return (has_value) ? 1 : 0;
            },
            pos,
            szs);
        ret.synchronize();
        timer.end();

        logger::raw_ln("impl =", cfg, "time =", timer.get_time_str());

        if (cfg == "reference") {
            ref_result = ret.copy_to_stdvec();
        } else {
            REQUIRE(compare<Tscal>(ref_result, ret.copy_to_stdvec(), 1e-12));
        }
    }
}

template<class Tscal>
inline void basic_histogram_size_non_unif(const std::vector<std::string> &impl_list) {

    logger::raw_ln(
        "testing basic_histogram_size_non_unif Tscal =", shambase::get_type_name<Tscal>());

    std::vector<Tscal> positions{};
    std::vector<Tscal> sizes{};

    size_t N      = 1e5;
    size_t Narray = 2048;

    std::mt19937_64 eng(0x42);
    for (size_t i = 0; i < N; i++) {
        auto tmp = shamalgs::primitives::mock_value<Tscal>(eng, 0, 1);
        positions.push_back(tmp * tmp);
        sizes.push_back(0.05);
    }

    std::vector<Tscal> arr_pos_inf{};
    std::vector<Tscal> arr_pos_sup{};
    for (size_t i = 0; i < Narray; i++) {
        arr_pos_inf.push_back(Tscal(i) / Tscal(Narray));
        arr_pos_sup.push_back(Tscal(i + 1) / Tscal(Narray));
    }

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceBuffer<Tscal> bin_inf(Narray, dev_sched);
    sham::DeviceBuffer<Tscal> bin_sup(Narray, dev_sched);
    sham::DeviceBuffer<Tscal> pos(N, dev_sched);
    sham::DeviceBuffer<Tscal> szs(N, dev_sched);

    bin_inf.copy_from_stdvec(arr_pos_inf);
    bin_sup.copy_from_stdvec(arr_pos_sup);
    pos.copy_from_stdvec(positions);
    szs.copy_from_stdvec(sizes);

    bin_inf.synchronize();
    bin_sup.synchronize();
    pos.synchronize();
    szs.synchronize();

    std::vector<Tscal> ref_result{};

    for (auto &cfg : impl_list) {
        using namespace shamalgs::primitives::impl;
        compute_histogram_impl_control.set_config(dev_sched, cfg);

        shambase::Timer timer;
        timer.start();
        sham::DeviceBuffer<Tscal> ret = shamalgs::primitives::compute_histogram<Tscal>(
            dev_sched,
            bin_inf,
            bin_sup,
            N,
            [](Tscal edge_inf, Tscal edge_sup, Tscal pos, Tscal sizes, bool &has_value) {
                has_value = edge_inf - sizes <= pos && pos < edge_sup + sizes;
                return (has_value) ? 1 : 0;
            },
            pos,
            szs);
        ret.synchronize();
        timer.end();

        logger::raw_ln("impl =", cfg, "time =", timer.get_time_str());

        if (cfg == "reference") {
            ref_result = ret.copy_to_stdvec();
        } else {
            REQUIRE(compare<Tscal>(ref_result, ret.copy_to_stdvec(), 1e-12));
        }
    }
}

TestStart(Unittest, "shamalgs::primitives::compute_histogram", shamalgsprimitivecomputehisto, 1) {

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    using namespace shamalgs::primitives::impl;

    auto impl_list = compute_histogram_impl_control.get_avail_configs(dev_sched);

    auto default_impl = compute_histogram_impl_control.get_default_config(dev_sched);

    basic_histogram<f32>(impl_list);
    basic_histogram<f64>(impl_list);
    basic_histogram_size<f32>(impl_list);
    basic_histogram_size<f64>(impl_list);
    basic_histogram_size_non_unif<f32>(impl_list);
    basic_histogram_size_non_unif<f64>(impl_list);

    compute_histogram_impl_control.set_config(dev_sched, default_impl);
}
