"""
Utility functions to download files
"""

import os
import sys
from urllib.request import urlretrieve

import shamrock.sys


def fmt(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024


def reporthook(block_num, block_size, total_size):
    if total_size <= 0 or block_size <= 0:
        return
    freq_report = int(int(total_size / block_size) / 10)
    if freq_report <= 0:
        return
    if block_num % freq_report == 0:
        BAR_WIDTH = 60
        downloaded = block_num * block_size
        percent = int((downloaded / total_size) * 100)
        filled = int(BAR_WIDTH * percent / 100)
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)
        sys.stdout.write(f"[{bar}] {percent:3d}% | {fmt(downloaded)}/{fmt(total_size)}\n")
        sys.stdout.flush()


def download_file(url, filename):
    """
    Download a file from an URL
    """

    if shamrock.sys.world_rank() == 0:
        print(f" - Downloading {filename} from {url}")
        # create the directory if it does not exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        urlretrieve(url, filename, reporthook=reporthook)

    shamrock.sys.mpi_barrier()

    # check that the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"File {filename} should have been downloaded but is not present on rank {shamrock.sys.world_rank()}"
        )
