import glob
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
    def __init__(self, model, dump_prefix, ext=".sham"):
        self.model = model
        self.dump_prefix = dump_prefix
        self.ext = ext
        os.makedirs(os.path.dirname(self.dump_prefix), exist_ok=True)

    def get_dump_name(self, idump):
        return self.dump_prefix + f"{idump:07}" + self.ext

    def get_last_dump(self):
        return get_last_dump(self.dump_prefix, self.ext)

    def purge_old_dumps(self, keep_first=1, keep_last=3):
        purge_old_dumps(self.dump_prefix, keep_first, keep_last, self.ext)

    def load_dump(self, idump):
        dump_name = self.get_dump_name(idump)
        if shamrock.sys.world_rank() == 0:
            print(f"Loading dump: {dump_name} i={idump}")
        self.model.load_from_dump(dump_name)

    def write_dump(self, idump, purge_old_dumps=False, keep_first=1, keep_last=3):
        dump_name = self.get_dump_name(idump)
        self.model.dump(dump_name)
        if purge_old_dumps:
            self.purge_old_dumps(keep_first, keep_last)

    def load_last_dump_or(self, functor_no_last_dump):
        idump = self.get_last_dump()
        if idump is None:
            return functor_no_last_dump()
        else:
            return self.load_dump(idump)
