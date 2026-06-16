import glob
import json
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
    Helper class to handle Shamrock checkpoint dump files.

    When ``metadata`` is enabled at construction, a JSON companion file is written
    and read alongside each checkpoint to store simulation metadata.
    """

    def __init__(self, model, dump_prefix, ext=".sham", metadata=False):
        """
        Parameters
        ----------
        model
            The Shamrock model instance used to write and load dumps.
        dump_prefix : str
            Path prefix for dump files; the dump index is appended as a
            zero-padded seven-digit number (e.g. ``prefix0000042``).
        ext : str, optional
            File extension for checkpoint dumps (default is ``".sham"``).
        metadata : bool, optional
            If ``True``, also write/read a ``.json`` companion with per-checkpoint
            metadata (default is ``False``).
        """
        self.model = model
        self.dump_prefix = dump_prefix
        self.ext = ext
        os.makedirs(os.path.dirname(self.dump_prefix), exist_ok=True)
        self.metadata = metadata

    def get_dump_name_extension(self, idump, ext) -> str:
        """Get the name of the dump file with the extension"""
        return self.dump_prefix + f"{idump:07}" + ext

    def get_dump_name(self, idump) -> str:
        """Get the name of the dump file (extension from self.ext)"""
        return self.get_dump_name_extension(idump, self.ext)

    def get_last_dump(self) -> int | None:
        """Find the last dump number.

        When metadata mode is enabled, validate that checkpoint dumps and JSON
        companion files agree on the latest checkpoint index.
        """
        last_dump = helper_get_last_dump(self.dump_prefix, self.ext)
        if not self.metadata:
            return last_dump

        last_metadata_dump = helper_get_last_dump(self.dump_prefix, ".json")
        if last_dump != last_metadata_dump:
            raise ValueError(
                "Detected inconsistent checkpoint files: "
                f"last {self.ext} dump is {last_dump}, "
                f"last .json dump is {last_metadata_dump}. "
                "This may indicate a botched checkpoint."
            )
        return last_dump

    def purge_old_dumps(self, keep_first=1, keep_last=3) -> None:
        """
        Purge old dump files.

        When metadata mode is enabled, also purge old JSON companion files.

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

        if self.metadata:
            helper_purge_old_dumps(self.dump_prefix, keep_first, keep_last, ".json")

    def load_dump(self, idump) -> dict | None:
        """
        Load a dump file.

        Parameters
        ----------
        idump : int
            The dump identifier to load.

        Returns
        -------
        dict or None
            If ``metadata`` was enabled at construction, the JSON metadata
            loaded from the companion file; otherwise ``None``.
        """
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

    def write_dump(self, idump, metadata=None, purge_old_dumps=False, keep_first=1, keep_last=3) -> None:
        """
        Write a dump file.

        Parameters
        ----------
        idump : int
            The dump identifier to write.
        metadata : object, optional
            JSON-serializable metadata stored in a ``.json`` companion next to the
            checkpoint. Required when ``metadata`` was enabled at construction.
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

        if self.metadata:
            if metadata is None:
                raise ValueError("metadata is required when metadata is enabled")

            if shamrock.sys.world_rank() == 0:
                with open(self.get_dump_name_extension(idump, ".json"), "w") as f:
                    json.dump(metadata, f)

        if purge_old_dumps:
            self.purge_old_dumps(keep_first, keep_last)

    def load_last_dump_or(self, functor_no_last_dump) -> dict | None:
        """
        Load the last dump or call a function if no dump is found.

        Parameters
        ----------
        functor_no_last_dump : callable
            Setup function invoked when no dump exists. Must not return a value.

        Returns
        -------
        dict or None
            If a dump was loaded and ``metadata`` was enabled at construction,
            the JSON metadata from the companion; otherwise ``None``.
        """
        idump = self.get_last_dump()
        if idump is None:
            result = functor_no_last_dump()
            if result is not None:
                raise ValueError("functor_no_last_dump must not return a value")
            return None
        else:
            return self.load_dump(idump)
