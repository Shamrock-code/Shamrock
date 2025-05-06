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
 * @file IDataEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shamdag/shamdag.hpp"

class IDataEdge {
    public:
    std::weak_ptr<INode> child;
    std::weak_ptr<INode> parent;

    template<class Func>
    inline void on_child(Func &&f) {
        if (auto spt = child.lock()) {
            f(shambase::get_check_ref(spt));
        } else {
            throw "";
        }
    }

    template<class Func>
    inline void on_parent(Func &&f) {
        if (auto spt = parent.lock()) {
            f(shambase::get_check_ref(spt));
        } else {
            throw "";
        }
    }

    template<class Func>
    inline void on_links(Func &&f) {
        auto spt_c = child.lock();
        auto spt_p = parent.lock();

        if (!bool(spt_c)) {
            throw "";
        }
        if (!bool(spt_p)) {
            throw "";
        }

        f(*spt_c, *spt_p);
    }

    void report_data_stealing();

    inline std::string get_label() { return _impl_get_label(); }

    inline std::string get_tex_symbol() { return _impl_get_tex_symbol(); }

    virtual std::string _impl_get_label()      = 0;
    virtual std::string _impl_get_tex_symbol() = 0;

    virtual ~IDataEdge() {}
};
