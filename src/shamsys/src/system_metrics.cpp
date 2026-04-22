// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file system_metrics.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/popen.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamcomm/local_rank.hpp"
#include "shamcomm/wrapper.hpp"
#include "shamsys/system_metrics.hpp"
#include <cstdlib>

#ifdef SHAMROCK_USE_GEOPM
    #include <geopm/PlatformIO.hpp>
    #include <geopm/PlatformTopo.hpp>
#endif

namespace shamsys {

#ifdef SHAMROCK_USE_GEOPM

    class AuroraSystemMetricReporterLinked : public ISystemMetricReporter {
        public:
        std::optional<f64> get_rank_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                return geopm::platform_io().read_signal("BOARD_ENERGY", GEOPM_DOMAIN_BOARD, 0);
            }
            return std::nullopt;
        }

        std::optional<f64> get_gpu_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                return geopm::platform_io().read_signal("GPU_ENERGY", GEOPM_DOMAIN_BOARD, 0);
            }
            return std::nullopt;
        }

        std::optional<f64> get_cpu_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                return geopm::platform_io().read_signal("CPU_ENERGY", GEOPM_DOMAIN_BOARD, 0);
            }
            return std::nullopt;
        }

        std::optional<f64> get_dram_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                return geopm::platform_io().read_signal("DRAM_ENERGY", GEOPM_DOMAIN_BOARD, 0);
            }
            return std::nullopt;
        }

        bool support_rank_energy_consummed() override { return true; }
        bool support_gpu_energy_consummed() override { return true; }
        bool support_cpu_energy_consummed() override { return true; }
        bool support_dram_energy_consummed() override { return true; }
    };
#endif

    class AuroraSystemMetricReporter : public ISystemMetricReporter {
        public:
        std::optional<f64> get_rank_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output("geopmread BOARD_ENERGY board 0");
                return std::stoull(output.c_str());
            }
            return std::nullopt;
        }

        std::optional<f64> get_gpu_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output("geopmread GPU_ENERGY board 0");
                return std::stoull(output.c_str());
            }
            return std::nullopt;
        }

        std::optional<f64> get_cpu_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output("geopmread CPU_ENERGY board 0");
                return std::stoull(output.c_str());
            }
            return std::nullopt;
        }

        std::optional<f64> get_dram_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output("geopmread DRAM_ENERGY board 0");
                return std::stoull(output.c_str());
            }
            return std::nullopt;
        }

        bool support_rank_energy_consummed() override { return true; }
        bool support_gpu_energy_consummed() override { return true; }
        bool support_cpu_energy_consummed() override { return true; }
        bool support_dram_energy_consummed() override { return true; }
    };

    class IntelRAPLSystemMetricReport : public ISystemMetricReporter {
        public:
        std::optional<f64> get_rank_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output(
                    "cat /sys/class/powercap/intel-rapl:0/energy_uj");
                return f64(std::stoull(output.c_str())) * 1e-6;
            }
            return std::nullopt;
        }

        std::optional<f64> get_gpu_energy_consummed() override { return std::nullopt; }

        std::optional<f64> get_cpu_energy_consummed() override { return std::nullopt; }

        std::optional<f64> get_dram_energy_consummed() override { return std::nullopt; }

        bool support_rank_energy_consummed() override { return true; }
        bool support_gpu_energy_consummed() override { return false; }
        bool support_cpu_energy_consummed() override { return false; }
        bool support_dram_energy_consummed() override { return false; }
    };

    class NoopSystemMetricReporter : public ISystemMetricReporter {
        public:
        std::optional<f64> get_rank_energy_consummed() override { return std::nullopt; }
        std::optional<f64> get_gpu_energy_consummed() override { return std::nullopt; }
        std::optional<f64> get_cpu_energy_consummed() override { return std::nullopt; }
        std::optional<f64> get_dram_energy_consummed() override { return std::nullopt; }

        bool support_rank_energy_consummed() override { return false; }
        bool support_gpu_energy_consummed() override { return false; }
        bool support_cpu_energy_consummed() override { return false; }
        bool support_dram_energy_consummed() override { return false; }
    };

    bool has_reporter() {
        auto &reporter = current_reporter();
        if (!reporter) {
            return false;
        }
        // dynamic_cast returns nullptr if the cast fails, so we check for that
        return dynamic_cast<NoopSystemMetricReporter *>(reporter.get()) == nullptr;
    }

    std::unique_ptr<ISystemMetricReporter> make_reporter(std::string_view reporter_name) {
        if (reporter_name == "aurora") {
            return std::make_unique<AuroraSystemMetricReporter>();
#ifdef SHAMROCK_USE_GEOPM
        } else if (reporter_name == "aurora-linked") {
            return std::make_unique<AuroraSystemMetricReporterLinked>();
#endif
        } else if (reporter_name == "intel-rapl") {
            return std::make_unique<IntelRAPLSystemMetricReport>();
        } else if (reporter_name == "noop" || reporter_name == "none" || reporter_name == "") {
            return std::make_unique<NoopSystemMetricReporter>();
        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "Unknown system metrics reporter: {}, valid reporters are: aurora, aurora-linked, "
                "intel-rapl, noop",
                reporter_name));
        }
        return std::make_unique<NoopSystemMetricReporter>();
    }

    std::unique_ptr<ISystemMetricReporter> make_reporter() {
        if (SHAM_SYSTEM_METRICS_REPORTER) {
            return make_reporter(*SHAM_SYSTEM_METRICS_REPORTER);
        }
        return std::make_unique<NoopSystemMetricReporter>();
    }

    /// test that there is no crashes
    void test_reporter(std::unique_ptr<ISystemMetricReporter> &reporter) {
        shambase::get_check_ref(reporter).get_rank_energy_consummed();
        shambase::get_check_ref(reporter).get_gpu_energy_consummed();
        shambase::get_check_ref(reporter).get_cpu_energy_consummed();
        shambase::get_check_ref(reporter).get_dram_energy_consummed();
    }

    std::unique_ptr<ISystemMetricReporter> &current_reporter() {
        static std::unique_ptr<ISystemMetricReporter> reporter = nullptr;
        if (!reporter) {
            reporter = make_reporter();
            test_reporter(reporter);
        }
        return reporter;
    }

    SystemMetrics get_system_metrics(bool barrier) {
        // Ensure that barriers aren't used if there is no reporter
        barrier = barrier && has_reporter();

        if (barrier) {
            shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        }
        f64 wall_time = shambase::details::get_wtime();
        auto ret      = SystemMetrics{
            wall_time,
            get_rank_energy_consummed(),
            get_gpu_energy_consummed(),
            get_cpu_energy_consummed(),
            get_dram_energy_consummed()};
        if (barrier) {
            shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        }
        return ret;
    }

    std::vector<SystemMetrics> gather_rank_metrics(const SystemMetrics &input) {
        std::vector<SystemMetrics> ret(shamcomm::world_size());

        auto optional_gather_power = [&](const std::optional<f64> &value) -> std::vector<f64> {
            return shamalgs::collective::gather(value ? value.value() : 0._f64);
        };

        std::vector<f64> rank_energy_consummed_all_ranks
            = optional_gather_power(input.rank_energy_consummed);
        std::vector<f64> gpu_energy_consummed_all_ranks
            = optional_gather_power(input.gpu_energy_consummed);
        std::vector<f64> cpu_energy_consummed_all_ranks
            = optional_gather_power(input.cpu_energy_consummed);
        std::vector<f64> dram_energy_consummed_all_ranks
            = optional_gather_power(input.dram_energy_consummed);
        std::vector<f64> metric_time_all_ranks = shamalgs::collective::gather(input.wall_time);

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            ret[i] = SystemMetrics{
                metric_time_all_ranks[i],
                (shamsys::support_rank_energy_consummed())
                    ? std::optional<f64>{rank_energy_consummed_all_ranks[i]}
                    : std::nullopt,
                (shamsys::support_gpu_energy_consummed())
                    ? std::optional<f64>{gpu_energy_consummed_all_ranks[i]}
                    : std::nullopt,
                (shamsys::support_cpu_energy_consummed())
                    ? std::optional<f64>{cpu_energy_consummed_all_ranks[i]}
                    : std::nullopt,
                (shamsys::support_dram_energy_consummed())
                    ? std::optional<f64>{dram_energy_consummed_all_ranks[i]}
                    : std::nullopt,
            };
        }

        return ret;
    }

    SystemMetrics aggregate_rank_metrics(const std::vector<SystemMetrics> &input) {
        f64 sum_rank_energy_consummed = 0._f64;
        f64 sum_gpu_energy_consummed  = 0._f64;
        f64 sum_cpu_energy_consummed  = 0._f64;
        f64 sum_dram_energy_consummed = 0._f64;
        f64 metric_time_all           = 0._f64;

        for (const auto &m : input) {
            sum_rank_energy_consummed
                += (m.rank_energy_consummed ? m.rank_energy_consummed.value() : 0._f64);
            sum_gpu_energy_consummed
                += (m.gpu_energy_consummed ? m.gpu_energy_consummed.value() : 0._f64);
            sum_cpu_energy_consummed
                += (m.cpu_energy_consummed ? m.cpu_energy_consummed.value() : 0._f64);
            sum_dram_energy_consummed
                += (m.dram_energy_consummed ? m.dram_energy_consummed.value() : 0._f64);
            metric_time_all = std::max(metric_time_all, m.wall_time);
        }

        SystemMetrics system_metrics;
        system_metrics.wall_time             = metric_time_all;
        system_metrics.rank_energy_consummed = (shamsys::support_rank_energy_consummed())
                                                   ? sum_rank_energy_consummed
                                                   : std::optional<f64>{};
        system_metrics.gpu_energy_consummed  = (shamsys::support_gpu_energy_consummed())
                                                   ? sum_gpu_energy_consummed
                                                   : std::optional<f64>{};
        system_metrics.cpu_energy_consummed  = (shamsys::support_cpu_energy_consummed())
                                                   ? sum_cpu_energy_consummed
                                                   : std::optional<f64>{};
        system_metrics.dram_energy_consummed = (shamsys::support_dram_energy_consummed())
                                                   ? sum_dram_energy_consummed
                                                   : std::optional<f64>{};

        return system_metrics;
    }

    FormattedSystemMetrics format_system_metrics(const SystemMetrics &input) {
        auto format_metric = [](const std::optional<f64> &energy,
                                f64 wall_time,
                                std::optional<std::string> &out_power,
                                std::optional<std::string> &out_energy) {
            if (energy.has_value()) {
                if (wall_time > 0._f64 && energy.value() > 0._f64) {
                    f64 consumed_energy = energy.value();
                    f64 power           = consumed_energy / wall_time;
                    out_power           = shambase::format("{:.1f} W", power);
                    out_energy          = shambase::format("{:.1f} J", consumed_energy);
                } else {
                    out_power  = "N/A";
                    out_energy = "N/A";
                }
            }
        };

        FormattedSystemMetrics ret{
            shambase::format("{:.1f} s", input.wall_time),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
        };

        format_metric(
            input.rank_energy_consummed,
            input.wall_time,
            ret.rank_power,
            ret.rank_energy_consummed);
        format_metric(
            input.gpu_energy_consummed,
            input.wall_time,
            ret.gpu_power,
            ret.gpu_energy_consummed /* */);
        format_metric(
            input.cpu_energy_consummed,
            input.wall_time,
            ret.cpu_power,
            ret.cpu_energy_consummed /* */);
        format_metric(
            input.dram_energy_consummed,
            input.wall_time,
            ret.dram_power,
            ret.dram_energy_consummed);

        return ret;
    }
} // namespace shamsys
