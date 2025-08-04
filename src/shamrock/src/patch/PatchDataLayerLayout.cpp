// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PatchDataLayerLayout.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shamrock/patch/PatchDataLayerLayout.hpp"

namespace shamrock::patch {
    std::string PatchDataLayerLayout::get_description_str() {
        std::stringstream ss;

        if (fields.empty()) {
            ss << "empty table\n";
        } else {

            u32 index = 0;
            for (var_t &v : fields) {
                v.visit([&](auto &field) {
                    using f_t    = typename std::remove_reference<decltype(field)>::type;
                    using base_t = typename f_t::field_T;

                    ss << index << " : " << field.name << " : nvar=" << field.nvar << " type : ";

                    if (std::is_same<base_t, f32>::value) {
                        ss << "f32   ";
                    } else if (std::is_same<base_t, f32_2>::value) {
                        ss << "f32_2 ";
                    } else if (std::is_same<base_t, f32_3>::value) {
                        ss << "f32_3 ";
                    } else if (std::is_same<base_t, f32_4>::value) {
                        ss << "f32_4 ";
                    } else if (std::is_same<base_t, f32_8>::value) {
                        ss << "f32_8 ";
                    } else if (std::is_same<base_t, f32_16>::value) {
                        ss << "f32_16";
                    } else if (std::is_same<base_t, f64>::value) {
                        ss << "f64   ";
                    } else if (std::is_same<base_t, f64_2>::value) {
                        ss << "f64_2 ";
                    } else if (std::is_same<base_t, f64_3>::value) {
                        ss << "f64_3 ";
                    } else if (std::is_same<base_t, f64_4>::value) {
                        ss << "f64_4 ";
                    } else if (std::is_same<base_t, f64_8>::value) {
                        ss << "f64_8 ";
                    } else if (std::is_same<base_t, f64_16>::value) {
                        ss << "f64_16";
                    } else if (std::is_same<base_t, u32>::value) {
                        ss << "u32   ";
                    } else if (std::is_same<base_t, u64>::value) {
                        ss << "u64   ";
                    } else if (std::is_same<base_t, u32_3>::value) {
                        ss << "u32_3 ";
                    } else if (std::is_same<base_t, u64_3>::value) {
                        ss << "u64_3 ";
                    } else {
                        ss << "unknown";
                    }

                    ss << "\n";

                    index++;
                });
            }
        }

        return ss.str();
    }

    std::vector<std::string> PatchDataLayerLayout::get_field_names() {
        std::vector<std::string> ret;

        for (var_t &v : fields) {
            v.visit([&](auto &field) {
                ret.push_back(field.name);
            });
        }

        return ret;
    }

    void to_json(nlohmann::json &j, const PatchDataLayerLayout &p) {

        using json = nlohmann::json;

        std::vector<json> entries;

        p.for_each_field_any([&](auto &field) {
            using f_t    = typename std::remove_reference<decltype(field)>::type;
            using base_t = typename f_t::field_T;

            auto get_tname = []() {
                if (std::is_same<base_t, f32>::value) {
                    return "f32";
                } else if (std::is_same<base_t, f32_2>::value) {
                    return "f32_2";
                } else if (std::is_same<base_t, f32_3>::value) {
                    return "f32_3";
                } else if (std::is_same<base_t, f32_4>::value) {
                    return "f32_4";
                } else if (std::is_same<base_t, f32_8>::value) {
                    return "f32_8";
                } else if (std::is_same<base_t, f32_16>::value) {
                    return "f32_16";
                } else if (std::is_same<base_t, f64>::value) {
                    return "f64";
                } else if (std::is_same<base_t, f64_2>::value) {
                    return "f64_2";
                } else if (std::is_same<base_t, f64_3>::value) {
                    return "f64_3";
                } else if (std::is_same<base_t, f64_4>::value) {
                    return "f64_4";
                } else if (std::is_same<base_t, f64_8>::value) {
                    return "f64_8";
                } else if (std::is_same<base_t, f64_16>::value) {
                    return "f64_16";
                } else if (std::is_same<base_t, u32>::value) {
                    return "u32";
                } else if (std::is_same<base_t, u64>::value) {
                    return "u64";
                } else if (std::is_same<base_t, u32_3>::value) {
                    return "u32_3";
                } else if (std::is_same<base_t, u64_3>::value) {
                    return "u64_3";
                } else if (std::is_same<base_t, i64_3>::value) {
                    return "i64_3";
                } else {
                    shambase::throw_unimplemented();
                    return "";
                }
            };

            entries.push_back(json{
                {"type", get_tname()},
                {"nvar", field.nvar},
                {"field_name", field.name},
            });
        });

        j = entries;
    }

    void from_json(const nlohmann::json &j, PatchDataLayerLayout &p) {
        for (auto &entry : j) {
            p.add_field_t(entry["field_name"], entry["nvar"].get<u32>(), entry["type"]);
        }
    }

    bool operator==(const PatchDataLayerLayout &lhs, const PatchDataLayerLayout &rhs) {

        bool ret = true;
        ret      = ret && (lhs.fields.size() == rhs.fields.size());

        for (u32 i = 0; i < lhs.fields.size(); i++) {
            const PatchDataLayerLayout::var_t &var_lhs = lhs.fields[i];
            const PatchDataLayerLayout::var_t &var_rhs = rhs.fields[i];

            std::visit(
                [&](auto &flhs, auto &frhs) {
                    using t1 = typename std::remove_reference<decltype(flhs)>::type;
                    using t2 = typename std::remove_reference<decltype(frhs)>::type;

                    ret = ret && std::is_same_v<t1, t2>;
                    ret = ret && (flhs.nvar == frhs.nvar);
                    ret = ret && (flhs.name == frhs.name);
                },
                var_lhs.value,
                var_rhs.value);
        }

        return ret;
    }

} // namespace shamrock::patch
