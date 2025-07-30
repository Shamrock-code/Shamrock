// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/random.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/math.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

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

TestStart(Unittest, "shamalgs/memory/SerializeHelper", test_serialize_helper, 1) {

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

TestStart(Benchmark, "shamalgs/memory/SerializeHelper:benchmark", bench_serializer, 1) {

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

        tser.end();

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

        tdeser.end();

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

        tser.end();

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

        tdeser.end();

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
