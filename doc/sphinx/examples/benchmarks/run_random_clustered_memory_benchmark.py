import random
from concurrent.futures import ThreadPoolExecutor

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


def make_index_constrained_shuffle(N, max_delta):
    idx = [i for i in range(N)]

    to_shuffle = [i for i in range(N)]
    # shuffle to_shuffle
    random.shuffle(to_shuffle)

    for i in to_shuffle:
        _imin = max(0, i - max_delta)
        _imax = min(N - 1, i + max_delta)

        j = random.randint(_imin, _imax)

        idx[i], idx[j] = idx[j], idx[i]

    return idx


Nelement = 2**25
cluster_sizes = [1, 2, 4, 8]
do_shuffle_options = [False, True]


def prepare_run(N, cluster_size, do_shuffle):
    print(f"Preparing run for N={N}, cluster_size={cluster_size}, do_shuffle={do_shuffle}")

    if N % cluster_size != 0:
        raise ValueError("N must be divisible by cluster_size")

    N_cluster = N // cluster_size

    to_shuffle = [i for i in range(N_cluster)]

    if do_shuffle:
        random.shuffle(to_shuffle)

    print(f"Prepared run for N={N}, cluster_size={cluster_size}, do_shuffle={do_shuffle}")

    return to_shuffle


runs = {}
# Create all parameter combinations
param_combinations = [
    (cluster_size, do_shuffle)
    for cluster_size in cluster_sizes
    for do_shuffle in do_shuffle_options
]

# Parallelize the prepare_run calls
with ThreadPoolExecutor() as executor:
    # Submit all jobs
    futures = {
        executor.submit(prepare_run, Nelement, cluster_size, do_shuffle): (cluster_size, do_shuffle)
        for cluster_size, do_shuffle in param_combinations
    }

    # Collect results
    for future in futures:
        cluster_size, do_shuffle = futures[future]
        runs[cluster_size, do_shuffle] = future.result()


def benchmark_random_chunk_copy(N, cluster_size, do_shuffle, shuffle_table, hardcoded=False):
    if N % cluster_size != 0:
        raise ValueError("N must be divisible by cluster_size")

    N_cluster = N // cluster_size
    buffer_source = shamrock.backends.DeviceBuffer_f64()
    buffer_index = shamrock.backends.DeviceBuffer_u32()
    buffer_result = shamrock.backends.DeviceBuffer_f64()

    buffer_source.resize(N)
    buffer_index.resize(N_cluster)
    buffer_result.resize(N)

    buffer_source.copy_from_stdvec([1.0 * i for i in range(N)])
    buffer_index.copy_from_stdvec(shuffle_table)
    buffer_result.copy_from_stdvec([0.0 for i in range(N)])

    if hardcoded:
        return shamrock.algs.benchmark_random_chunk_copy_hardcoded(
            buffer_source, buffer_index, cluster_size, buffer_result
        )
    else:
        return shamrock.algs.benchmark_random_chunk_copy(
            buffer_source, buffer_index, cluster_size, buffer_result
        )


for cluster_size in cluster_sizes:
    for do_shuffle in do_shuffle_options:
        N = Nelement
        t = benchmark_random_chunk_copy(
            N, cluster_size, do_shuffle, runs[cluster_size, do_shuffle], hardcoded=False
        )
        print(
            f"N={N}, cluster_size={cluster_size}, do_shuffle={do_shuffle}, bandwidth={t:e} (non-hardcoded)"
        )
        t = benchmark_random_chunk_copy(
            N, cluster_size, do_shuffle, runs[cluster_size, do_shuffle], hardcoded=True
        )
        print(
            f"N={N}, cluster_size={cluster_size}, do_shuffle={do_shuffle}, bandwidth={t:e} (hardcoded)"
        )
