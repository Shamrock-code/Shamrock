import glob
import json
import os

import shamrock.sys


def purge_old_dumps(dump_prefix, keep_first=1, keep_last=3, ext=".sham"):
    if shamrock.sys.world_rank() == 0:
        res = glob.glob(dump_prefix + "*" + ext)
        res.sort()

        # The list of dumps to remove (keep the first and last 3 dumps)
        to_remove = res[keep_first:-keep_last]

        for f in to_remove:
            os.remove(f)


def get_last_dump(dump_prefix, ext=".sham"):
    res = glob.glob(dump_prefix + "*" + ext)

    num_max = -1

    for f in res:
        try:
            dump_num = int(f[len(dump_prefix) : -len(ext)])
            if dump_num > num_max:
                num_max = dump_num
        except ValueError:
            pass

    if num_max == -1:
        return None
    else:
        return num_max


class ShamrockDumpHandleHelper:
    def __init__(self, model, dump_prefix, ext=".sham", metadata=False):
        self.model = model
        self.dump_prefix = dump_prefix
        self.ext = ext
        os.makedirs(os.path.dirname(self.dump_prefix), exist_ok=True)
        self.metadata = metadata

    def get_dump_name_extension(self, idump, ext):
        return self.dump_prefix + f"{idump:07}" + ext

    def get_dump_name(self, idump):
        return self.get_dump_name_extension(idump, self.ext)

    def get_last_dump(self):
        return get_last_dump(self.dump_prefix, self.ext)

    def purge_old_dumps(self, keep_first=1, keep_last=3):
        purge_old_dumps(self.dump_prefix, keep_first, keep_last, self.ext)

    def load_dump(self, idump):
        dump_name = self.get_dump_name(idump)
        if shamrock.sys.world_rank() == 0:
            print(f"Loading dump: {dump_name} i={idump}")
        self.model.load_from_dump(dump_name)
        if self.metadata:
            dump_name = self.get_dump_name_extension(idump, ".json")
            with open(dump_name, "r") as f:
                return json.load(f)
        else:
            return None

    def write_dump(self, idump, metadata=None, purge_old_dumps=False, keep_first=1, keep_last=3):
        dump_name = self.get_dump_name(idump)
        self.model.dump(dump_name)

        if self.metadata and shamrock.sys.world_rank() == 0:
            if metadata is None:
                raise ValueError("metadata is required when metadata is enabled")

            with open(self.get_dump_name_extension(idump, ".json"), "w") as f:
                json.dump(metadata, f)

        if purge_old_dumps:
            self.purge_old_dumps(keep_first, keep_last)

    def load_last_dump_or(self, functor_no_last_dump):
        idump = self.get_last_dump()
        if idump is None:
            functor_no_last_dump()
            return None
        else:
            return self.load_dump(idump)
