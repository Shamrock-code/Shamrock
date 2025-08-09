// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/details/random/random.hpp"
#include "shamrock/scheduler/loadbalance/LoadBalanceStrategy.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"

using namespace shamrock::scheduler;
using Tweight = u64;
using Torder  = u64;
using LBTile  = TileWithLoad<Torder, Tweight>;

void add_strategy_plot(
    std::string strat_name,
    std::string filename,
    const std::vector<LBTile> &vec_test,
    const std::vector<i32> &result,
    i32 wsize) {

    std::vector<f64> load_values;
    std::vector<f64> order_values;
    std::vector<i32> node_owner;

    for (u32 i = 0; i < vec_test.size(); i++) {
        load_values.push_back(f64(vec_test[i].load_value));
        order_values.push_back(f64(vec_test[i].ordering_val));
        node_owner.push_back(result[i]);
    }

    PyScriptHandle hndl{};

    hndl.data()["wsize"]      = wsize;
    hndl.data()["loads"]      = load_values;
    hndl.data()["order"]      = order_values;
    hndl.data()["node_owner"] = node_owner;
    hndl.data()["filename"]   = std::string("tests/figures/load_balance_strat") + filename + ".pdf";
    hndl.data()["strat_name"] = strat_name;

    hndl.exec(R"py(

        import matplotlib.pyplot as plt
        import numpy as np

        plt.close('all')

        range_wsize = range(wsize)

        ptch_lst_node = [[] for i in range_wsize]

        for load,ord,own in zip(loads, order, node_owner):

            ptch_lst_node[own].append(load)

        lens = [len(i) for i in ptch_lst_node]
        mx = max(lens)

        for i in ptch_lst_node:
            while len(i) < mx:
                i.append(0)


        bar_lst = np.transpose(ptch_lst_node)

        cummul = np.array([0. for j in range_wsize])

        for bindex in range(len(bar_lst)):
            i = bar_lst[bindex]
            plt.bar(range(len(i)), i, bottom=cummul)

            cummul += np.array(i)

        plt.xlabel("nodes id")
        plt.ylabel("load")
        plt.title(strat_name)

        plt.savefig(filename)

        plt.close('all')

    )py");

    TEX_REPORT(
        R"tex(

        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.95\linewidth]{figures/load_balance_strat)tex"
        + filename + R"tex(.pdf}
        \caption{Load balancing strategy}
        \end{figure}

    )tex")
}

TestStart(TestType::ValidationTest, "shamrock/scheduler/loadbalance", testloadbalancestrat, 1) {

    i32 fake_world_size = 64;

    auto make_tile_list = [](u32 count, u64 min_load, u64 max_load) -> std::vector<LBTile> {
        std::vector<LBTile> res;
        std::mt19937 eng{0x111};

        for (u32 i = 0; i < count; i++) {
            res.push_back(LBTile{
                shamalgs::primitives::mock_value(eng, 0_u64, u64_max),
                shamalgs::primitives::mock_value(eng, min_load, max_load),
            });
        }

        return res;
    };

    std::vector<LBTile> vec_test = make_tile_list(64 * 4, 1000000, 1200000);

    std::vector<i32> result1     = details::lb_startegy_parallel_sweep(vec_test, fake_world_size);
    std::vector<i32> result2     = details::lb_startegy_roundrobin(vec_test, fake_world_size);
    std::vector<i32> result_best = load_balance(std::vector(vec_test), fake_world_size);

    add_strategy_plot("parallel sweep", "psweep", vec_test, result1, fake_world_size);
    add_strategy_plot("round robin", "rrobin", vec_test, result2, fake_world_size);
    add_strategy_plot("best", "rrobin", vec_test, result_best, fake_world_size);
}
