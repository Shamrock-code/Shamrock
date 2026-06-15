import glob
import os

import shamrock.sys


def helper_purge_old_dumps(dump_prefix, keep_first=1, keep_last=3, ext=".sham") -> None:
    """
    Purge old dump files.
    """
    if shamrock.sys.world_rank() == 0:
        res = glob.glob(dump_prefix + "*" + ext)
        res.sort()

        # The list of dumps to remove (keep the first and last 3 dumps)
        to_remove = res[keep_first:-keep_last]

        for f in to_remove:
            os.remove(f)


def helper_get_last_dump(dump_prefix, ext=".sham") -> int | None:
    """
    Get the last dump number.
    """
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
    """
    Helper class to handle dump files.
    """

    def __init__(self, model, dump_prefix, ext=".sham"):
        self.model = model
        self.dump_prefix = dump_prefix
        self.ext = ext
        os.makedirs(os.path.dirname(self.dump_prefix), exist_ok=True)

    def get_dump_name_extension(self, idump, ext) -> str:
        """Get the name of the dump file with the extension"""
        return self.dump_prefix + f"{idump:07}" + ext

    def get_dump_name(self, idump) -> str:
        """Get the name of the dump file (extension from self.ext)"""
        return self.get_dump_name_extension(idump, self.ext)

    def get_last_dump(self) -> int | None:
        """Find the last dump number"""
        return helper_get_last_dump(self.dump_prefix, self.ext)

    def purge_old_dumps(self, keep_first=1, keep_last=3) -> None:
        """
        Purge old dump files.

        Parameters
        ----------
        keep_first : int, optional
            Number of oldest dump files to keep (default is 1, i.e. keep the first dump).
        keep_last : int, optional
            Number of newest dump files to keep (default is 3, i.e. keep the last 3 dumps).

        Returns
        -------
        None
            This method does not return a value.
        """
        helper_purge_old_dumps(self.dump_prefix, keep_first, keep_last, self.ext)

    def load_dump(self, idump) -> None:
        """
        Load a dump file.

        Parameters
        ----------
        idump : int
            The dump identifier to load.

        Returns
        -------
        None
            This method does not return a value.
        """
        dump_name = self.get_dump_name(idump)
        if shamrock.sys.world_rank() == 0:
            print(f"Loading dump: {dump_name} i={idump}")
        self.model.load_from_dump(dump_name)

    def write_dump(self, idump, purge_old_dumps=False, keep_first=1, keep_last=3) -> None:
        """
        Write a dump file.

        Parameters
        ----------
        idump : int
            The dump identifier to write.
        purge_old_dumps : bool, optional
            Whether to purge old dumps (default is False).
        keep_first : int, optional
            Number of oldest dump files to keep (default is 1, i.e. keep the first dump).
        keep_last : int, optional
            Number of newest dump files to keep (default is 3, i.e. keep the last 3 dumps).

        Returns
        -------
        None
            This method does not return a value.
        """
        dump_name = self.get_dump_name(idump)
        self.model.dump(dump_name)
        if purge_old_dumps:
            self.purge_old_dumps(keep_first, keep_last)

    def load_last_dump_or(self, functor_no_last_dump) -> None:
        """
        Load the last dump or call a function if no dump is found.

        Parameters
        ----------
        functor_no_last_dump : callable
            The function to call if no dump are found (i.e. the setup function).
        """
        idump = self.get_last_dump()
        if idump is None:
            result = functor_no_last_dump()
            if result is not None:
                raise ValueError("functor_no_last_dump must not return a value")
        else:
            self.load_dump(idump)
