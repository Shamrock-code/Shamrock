// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shamalgs/random.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/math.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace {

    std::vector<u8> mock_vector_tiled(u64 seed, size_t len) {
        constexpr size_t mock_vector_tile_size = 65536;
        static_assert(mock_vector_tile_size <= std::numeric_limits<u32>::max());

        const size_t tile_len = mock_vector_tile_size;
        std::vector<u8> tile
            = shamalgs::primitives::mock_vector<u8>(seed, tile_len, u8{0}, u8{255});

        std::vector<u8> buf(len);
        for (size_t off = 0; off < len; off += tile_len) {
            const size_t n = std::min<size_t>(tile_len, len - off);
            std::memcpy(buf.data() + off, tile.data(), n);
        }
        return buf;
    }

} // namespace

template<class T>
inline void check_buf(std::string prefix, sycl::buffer<T> &b1, sycl::buffer<T> &b2) {

    REQUIRE_EQUAL_NAMED(prefix + std::string("same size"), b1.size(), b2.size());

    {
        sycl::host_accessor acc1{b1};
        sycl::host_accessor acc2{b2};

        std::string id_err_list = "errors in id : ";

        bool eq = true;
        for (u32 i = 0; i < b1.size(); i++) {
            if (!sham::equals(acc1[i], acc2[i])) {
                eq = false;
                // id_err_list += std::to_string(i) + " ";
            }
        }

        if (eq) {
            REQUIRE_NAMED("same content", eq);
        } else {
            shamtest::asserts().assert_add_comment("same content", eq, id_err_list);
        }
    }
}

NEW_TEST(Unittest, "shamalgs/memory/SerializeHelper", 1) {

    u32 n1                     = 100;
    sycl::buffer<u8> buf_comp1 = shamalgs::random::mock_buffer<u8>(0x111, n1);

    f64_16 test_val = f64_16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    std::string test_str = "physics phd they said";

    u32 n2                        = 100;
    sycl::buffer<u32_3> buf_comp2 = shamalgs::random::mock_buffer<u32_3>(0x121, n2);

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());

    shamalgs::SerializeSize bytelen
        = ser.serialize_byte_size<u8>(n1) + ser.serialize_byte_size<f64_16>()
          + ser.serialize_byte_size<u32_3>(n2) + ser.serialize_byte_size(test_str);

    ser.allocate(bytelen);
    ser.write_buf(buf_comp1, n1);
    ser.write(test_val);
    ser.write(test_str);
    ser.write_buf(buf_comp2, n2);

    logger::raw_ln("writing done");

    auto recov = ser.finalize();

    {
        sycl::buffer<u8> buf1(n1);
        f64_16 val;
        std::string recv_str;
        sycl::buffer<u32_3> buf2(n2);

        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        logger::raw_ln("load 1 ");
        ser2.load_buf(buf1, n1);
        logger::raw_ln("load 1 done");
        ser2.load(val);
        logger::raw_ln("load 2 done");
        ser2.load(recv_str);
        logger::raw_ln("load 3 done");
        ser2.load_buf(buf2, n2);
        logger::raw_ln("load 4 done");

        // shamalgs::memory::print_buf(buf_comp1, n1, 16, "{} ");
        // shamalgs::memory::print_buf(buf1, n1, 16, "{} ");

        REQUIRE_NAMED("same", sham::equals(val, test_val));
        REQUIRE_NAMED("same", test_str == recv_str);
        check_buf("buf 1", buf_comp1, buf1);
        check_buf("buf 2", buf_comp2, buf2);
    }
}

NEW_TEST(Unittest, "shamalgs/memory/SerializeHelper/large_buffer", 1) {

    // TODO: find a way to do 4GB but Github CI is too small
    constexpr size_t buf_len = 600'000'000ull;

    std::vector<u8> ref;
    try {
        ref = mock_vector_tiled(0x1111, buf_len);
    } catch (const std::bad_alloc &) {
        REQUIRE(false);
        return;
    }

    auto sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q    = shambase::get_check_ref(sched).get_queue().q;

    sycl::buffer<u8> buf(ref.data(), buf_len);

    shamalgs::SerializeHelper ser(sched);
    shamalgs::SerializeSize sz = shamalgs::SerializeHelper::serialize_byte_size<u8>(buf_len);
    ser.allocate(sz, true);
    ser.write_buf(buf, buf_len);
    auto recov = ser.finalize();
    q.wait();

    sham::DeviceBuffer<u8> buf_out(buf_len, sched);
    shamalgs::SerializeHelper ser2(sched, std::move(recov), true);
    ser2.load_buf(buf_out, buf_len);
    q.wait();

    std::vector<u8> got = buf_out.copy_to_stdvec();
    REQUIRE_EQUAL(ref, got);
}

NEW_TEST(Unittest, "shamalgs/memory/SerializeHelper/large_buffer_narrowing", 1) {

    // Total serialized byte length must exceed u32_max for narrow_or_throw<u32> to fail.
    constexpr size_t buf_len = 4'300'000'000ull;

    auto sched                 = shamsys::instance::get_compute_scheduler_ptr();
    shamalgs::SerializeSize sz = shamalgs::SerializeHelper::serialize_byte_size<u8>(buf_len);

    shamalgs::SerializeHelper ser(sched);
    REQUIRE_EXCEPTION_THROW(ser.allocate(sz), std::runtime_error);

    sham::DeviceBuffer<u8> storage(buf_len, sched);
    REQUIRE_EXCEPTION_THROW(
        shamalgs::SerializeHelper(sched, std::move(storage), false), std::runtime_error);
}

NEW_TEST(Benchmark, "shamalgs/memory/SerializeHelper:benchmark", 1) {

    auto get_perf_knownsize = [](u32 buf_cnt, u32 buf_len) -> std::pair<f64, f64> {
        StackEntry stack{};
        std::vector<sycl::buffer<f64>> bufs;
        std::vector<sycl::buffer<f64>> bufs_ret;

        for (u32 i = 0; i < buf_cnt; i++) {
            bufs.emplace_back(shamalgs::random::mock_buffer<f64>(0x111 + i, buf_len));
        }

        shambase::Timer tser;
        tser.start();

        shamalgs::SerializeHelper ser1(shamsys::instance::get_compute_scheduler_ptr());
        shamalgs::SerializeSize sz = ser1.serialize_byte_size<f64>(buf_cnt * buf_len);
        ser1.allocate(sz);
        for (u32 i = 0; i < buf_cnt; i++) {
            ser1.write_buf(bufs[i], buf_len);
        }
        auto recov = ser1.finalize();
        shamsys::instance::get_compute_queue().wait();

        tser.stop();

        shambase::Timer tdeser;
        tdeser.start();

        for (u32 i = 0; i < buf_cnt; i++) {
            bufs_ret.emplace_back(sycl::buffer<f64>(buf_len));
        }

        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
        for (u32 i = 0; i < buf_cnt; i++) {
            ser2.load_buf(bufs_ret[i], buf_len);
        }
        shamsys::instance::get_compute_queue().wait();

        tdeser.stop();

        return {
            sz.get_total_size() / (tser.nanosec / 1e9),
            sz.get_total_size() / (tdeser.nanosec / 1e9)};
    };
    auto get_perf_unknownsize = [](u32 buf_cnt, u32 buf_len) -> std::pair<f64, f64> {
        StackEntry stack{};
        std::vector<sycl::buffer<f64>> bufs;
        std::vector<sycl::buffer<f64>> bufs_ret;

        for (u32 i = 0; i < buf_cnt; i++) {
            bufs.emplace_back(shamalgs::random::mock_buffer<f64>(0x111 + i, buf_len));
        }

        shambase::Timer tser;
        tser.start();

        shamalgs::SerializeHelper ser1(shamsys::instance::get_compute_scheduler_ptr());
        shamalgs::SerializeSize sz = ser1.serialize_byte_size<f64>(buf_cnt * buf_len)
                                     + (ser1.serialize_byte_size<u32>() * buf_cnt);
        ser1.allocate(sz);
        for (u32 i = 0; i < buf_cnt; i++) {
            ser1.write(buf_len);
            ser1.write_buf(bufs[i], buf_len);
        }
        auto recov = ser1.finalize();
        shamsys::instance::get_compute_queue().wait();

        tser.stop();

        shambase::Timer tdeser;
        tdeser.start();

        for (u32 i = 0; i < buf_cnt; i++) {
            bufs_ret.emplace_back(sycl::buffer<f64>(buf_len));
        }

        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
        for (u32 i = 0; i < buf_cnt; i++) {
            u32 tmp;
            ser2.load(tmp);
            ser2.load_buf(bufs_ret[i], buf_len);
        }
        shamsys::instance::get_compute_queue().wait();

        tdeser.stop();

        return {
            sz.get_total_size() / (tser.nanosec / 1e9),
            sz.get_total_size() / (tdeser.nanosec / 1e9)};
    };
    {
        std::vector<f64> x;
        std::vector<f64> tser, tdeser;
        std::vector<f64> tser_usz, tdeser_usz;

        for (u32 i = 1; i < 10000; i *= 2) {
            shamlog_debug_ln("Test", "i =", i);

            auto [p1, p2] = get_perf_knownsize(i, 100);
            auto [p3, p4] = get_perf_unknownsize(i, 100);
            x.push_back(i);
            tser.push_back(p1);
            tdeser.push_back(p2);
            tser_usz.push_back(p3);
            tdeser_usz.push_back(p4);
        }

        PyScriptHandle hdnl{};

        hdnl.data()["X"]         = x;
        hdnl.data()["ser_ksz"]   = tser;
        hdnl.data()["deser_ksz"] = tdeser;
        hdnl.data()["ser_usz"]   = tser_usz;
        hdnl.data()["deser_usz"] = tdeser_usz;

        hdnl.exec(R"py(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(15,6))

        axs.plot(X,ser_ksz,'-',c = 'black',label = "serialize (kz)")
        axs.plot(X,deser_ksz,':',c = 'black',label = "deserialize (kz)")
        axs.plot(X,ser_usz,'--',c = 'black',label = "serialize (ukz)")
        axs.plot(X,deser_usz,'-.',c = 'black',label = "deserialize (ukz)")

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('buffer count')
        plt.ylabel('Bandwidth (B.s-1)')
        plt.legend()
        plt.tight_layout()

        plt.savefig("tests/figures/benchmark-serialize1.pdf")

    )py");

        TEX_REPORT(R"==(

        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.95\linewidth]{tests/figures/benchmark-serialize1}
        \caption{Test the serializehelper performance (buf size = 100 f64)}
        \end{figure}

    )==")
    }

    {
        std::vector<f64> x;
        std::vector<f64> tser, tdeser;
        std::vector<f64> tser_usz, tdeser_usz;

        for (u32 i = 8; i < 10000; i *= 2) {
            shamlog_debug_ln("Test", "i =", i);

            auto [p1, p2] = get_perf_knownsize(1000, i);
            auto [p3, p4] = get_perf_unknownsize(1000, i);
            x.push_back(i);
            tser.push_back(p1);
            tdeser.push_back(p2);
            tser_usz.push_back(p3);
            tdeser_usz.push_back(p4);
        }

        PyScriptHandle hdnl{};

        hdnl.data()["X"]         = x;
        hdnl.data()["ser_ksz"]   = tser;
        hdnl.data()["deser_ksz"] = tdeser;
        hdnl.data()["ser_usz"]   = tser_usz;
        hdnl.data()["deser_usz"] = tdeser_usz;

        hdnl.exec(R"py(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(15,6))

        axs.plot(X,ser_ksz,'-',c = 'black',label = "serialize (kz)")
        axs.plot(X,deser_ksz,':',c = 'black',label = "deserialize (kz)")
        axs.plot(X,ser_usz,'--',c = 'black',label = "serialize (ukz)")
        axs.plot(X,deser_usz,'-.',c = 'black',label = "deserialize (ukz)")

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('buffer size (f64)')
        plt.ylabel('Bandwidth (B.s-1)')
        plt.legend()
        plt.tight_layout()

        plt.savefig("tests/figures/benchmark-serialize2.pdf")

    )py");

        TEX_REPORT(R"==(

        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.95\linewidth]{tests/figures/benchmark-serialize2}
        \caption{Test the serializehelper performance (buf count = 1000)}
        \end{figure}

    )==")
    }
}
