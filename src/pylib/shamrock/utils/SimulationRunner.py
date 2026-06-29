import types
from dataclasses import dataclass
from math import inf
from typing import Callable

import shamrock
from shamrock.utils.dump import ShamrockDumpHandleHelper

# ----------------------------
# decorators
# ----------------------------


def callback(*, tsim_interval=None, iter_count_interval=None, walltime_interval=None):
    """
    Decorator to mark a function as a simulation callback.

    Example:
        @callback(tsim_interval=1.0)
        def analysis(self, icallback):
            print("analysis", icallback)

    Args:
        tsim_interval: The time step of the callback.
        iter_count_interval: The iteration count interval of the callback.
        walltime_interval: The walltime interval of the callback.
    Returns:
        The decorated function.
    """

    # at least one of the intervals must be provided
    if tsim_interval is None and iter_count_interval is None and walltime_interval is None:
        raise ValueError("At least one of the intervals must be provided")

    def deco(func):
        func.__simulation_callback__ = {
            "func_name": func.__name__,
            "tsim_interval": tsim_interval,
            "iter_count_interval": iter_count_interval,
            "walltime_interval": walltime_interval,
        }

        return func

    return deco


def simulation_setup(func):
    """
    Decorator to mark a function as a simulation setup.

    Example:
        @simulation_setup
        def setup(self):
            print("setup")
    """

    func.__simulation_setup__ = True
    return func


# ----------------------------
# metaclass
# ----------------------------


class SimulationMeta(type):
    def __new__(mcls, name, bases, namespace):

        cls = super().__new__(mcls, name, bases, namespace)

        # ----------------------------
        # verbosity flag (default False)
        # Just add __debug_class_creation__ = True to a derived class to enable verbose mode
        # ----------------------------
        verbose = namespace.get("__debug_class_creation__", False)

        if verbose:
            print("\n==============================")
            print(f"[metaclass] Creating class: {name}")
            print("==============================\n")

            print("=== RAW NAMESPACE ===")
            for k, v in namespace.items():
                print(f"{k:25} {type(v)}")
            print()

        # skip base class
        if name == "SimulationRunner":
            return cls

        callbacks = []
        setup_func = None

        if verbose:
            print("=== INSPECTION ===")

        for name, obj in namespace.items():
            if isinstance(obj, (types.FunctionType, classmethod, staticmethod)):
                if isinstance(obj, (classmethod, staticmethod)):
                    func = obj.__func__
                else:
                    func = obj

                cb = getattr(func, "__simulation_callback__", None)
                setup = getattr(func, "__simulation_setup__", None)

                if verbose:
                    if cb is not None:
                        print(f"[decorator callback] applying to: {name} | value: {cb}")
                    if setup is not None:
                        print(f"[decorator setup] applying to: {name} | value: {setup}")

                if cb is not None:
                    callbacks.append((name, cb))

                if setup:
                    if setup_func is not None:
                        raise ValueError("Multiple setup functions")

                    setup_func = (name, func)

        if verbose:
            print("\n=== Metaclass result ===")
            print("callbacks:", callbacks)
            print("setup_func:", setup_func)

        if setup_func is None:
            raise ValueError("No simulation setup function found")

        cls._declared_callbacks = callbacks
        cls._setup = setup_func

        return cls


# ----------------------------
# base class
# ----------------------------
def rank_0_print(*args, **kwargs):
    if shamrock.sys.world_rank() == 0:
        print(*args, **kwargs)


@dataclass
class CallbackInfo:
    func: Callable
    name: str

    tsim_interval: float | None = None
    iter_count_interval: int | None = None
    walltime_interval: float | None = None


class CallbackState:
    def __init__(self, info: CallbackInfo, tsim_start: float):
        self.info = info
        self.counter = 0
        self.next_tsim = tsim_start if info.tsim_interval is not None else None
        self.next_iter_count = 0 if info.iter_count_interval is not None else None
        self.next_walltime = 0.0 if info.walltime_interval is not None else None

    def advance(self, t_model: float, iter_count: int, walltime: float):
        self.counter += 1

        if self.info.tsim_interval is not None:
            self.next_tsim = t_model + self.info.tsim_interval
        if self.info.iter_count_interval is not None:
            self.next_iter_count = iter_count + self.info.iter_count_interval
        if self.info.walltime_interval is not None:
            self.next_walltime = walltime + self.info.walltime_interval

        rank_0_print(f'[Simulation] Advancing callback "{self.info.name}"')
        if self.info.tsim_interval is not None:
            rank_0_print(f"   -> t = {t_model} -> {self.next_tsim}")
        if self.info.iter_count_interval is not None:
            rank_0_print(f"   -> iter = {iter_count} -> {self.next_iter_count}")
        if self.info.walltime_interval is not None:
            rank_0_print(f"   -> walltime = {walltime} -> {self.next_walltime}")

    def should_trigger(self, t_model: float, iter_count: int, walltime: float) -> bool:
        trig = False

        log = []

        if self.info.tsim_interval is not None:
            if t_model >= self.next_tsim:  # should i add a tolerance here ?
                trig = True
                log.append(f"   -> t = {t_model} >= {self.next_tsim}")
        if self.info.iter_count_interval is not None:
            if iter_count >= self.next_iter_count:
                trig = True
                log.append(f"   -> iter = {iter_count} >= {self.next_iter_count}")
        if self.info.walltime_interval is not None:
            if walltime >= self.next_walltime:
                trig = True
                log.append(f"   -> walltime = {walltime} >= {self.next_walltime}")

        if trig:
            rank_0_print(
                f'[Simulation] Triggering callback "{self.info.name}" (counter = {self.counter}):\n'
                + "\n".join(log)
            )

        return trig

    def to_dict(self):
        return {
            "counter": self.counter,
            "next_tsim": self.next_tsim,
            "next_iter_count": self.next_iter_count,
            "next_walltime": self.next_walltime,
        }

    def from_dict(self, data: dict):
        self.counter = data["counter"]
        self.next_tsim = data["next_tsim"]
        self.next_iter_count = data["next_iter_count"]
        self.next_walltime = data["next_walltime"]


class SimulationRunner(metaclass=SimulationMeta):
    """
    SimulationRunner is a base class to declare a simulation with setup & callbacks.

    A derived class must define:
    - t_end: float = <end time of the simulation>
    - a setup (any function decorated with @simulation_setup)

    And can define callbacks (any function decorated with @callback):

    < call every tsim = i * time_step >
    - @callback(time_step=1.0)
      def analysis(self, icallback):
          rank_0_print("analysis")

    < call when tsim = dt_stop, niter_max is reached or walltime_step is reached >
    - @callback(time_step=dt_stop, niter_max=1000, walltime_step=30*60)
      def do_checkpoint(self, icheckpoint):
          self.dump_helper.dump(icheckpoint)

    Note that for the last one that this reset the counters until next callback.
    The trigger conditions are inclusive and reset the counters for all triggers of that callback.
    """

    t_end: float | None = None
    dump_prefix: str | None = None

    cur_t: float = 0.0
    cur_iter_count: int = 0

    _declared_callbacks: list  # Will be filled by the metaclass
    _setup: tuple[str, Callable]  # Will be filled by the metaclass

    def __init__(self, model):
        self.model = model

        self._callbacks = []

        for name, info in self._declared_callbacks:
            copied = CallbackInfo(
                func=getattr(self, name),
                name=name,
                tsim_interval=info["tsim_interval"],
                iter_count_interval=info["iter_count_interval"],
                walltime_interval=info["walltime_interval"],
            )

            self._callbacks.append(copied)

        self._callbacks_state = None

        if self.dump_prefix is not None:
            self.dump_helper = ShamrockDumpHandleHelper(self.model, self.dump_prefix, metadata=True)
        else:
            self.dump_helper = None

        if self.t_end is None:
            raise ValueError(f"{type(self).__name__}.t_end must be defined")

        if self._declared_callbacks is None:
            raise ValueError(f"{type(self).__name__}._declared_callbacks must be defined")

        if self._setup is None:
            raise ValueError(f"{type(self).__name__}._setup must be defined")

    def do_checkpoint(self, icheckpoint: int, **kwargs):

        if self.dump_prefix is None:
            raise ValueError(f"{type(self).__name__}.dump_prefix must be defined")

        metadata = {
            "cur_t": self.cur_t,
            "cur_iter_count": self.cur_iter_count,
        }

        for ic, c in enumerate(self._callbacks):
            metadata[c.name] = self._callbacks_state[ic].to_dict()

        rank_0_print("[Simulation] Doing checkpoint")
        self.dump_helper.write_dump(icheckpoint, metadata=metadata, **kwargs)
        rank_0_print("[Simulation] Checkpoint done")

    def run_setup(self):

        name, func = self._setup
        rank_0_print()
        rank_0_print(f"[Simulation] Running setup function: {name}")
        rank_0_print()
        func(self)

        self.cur_t = self.model.get_time()
        self.cur_iter_count = 0

        rank_0_print()
        rank_0_print("[Simulation] Setting up callbacks states")
        self._callbacks_state = [CallbackState(c, self.cur_t) for c in self._callbacks]

        rank_0_print("[Simulation] Setup done")

    def restore_from_checkpoint(self, metadata: dict):
        self.cur_t = metadata["cur_t"]
        self.cur_iter_count = metadata["cur_iter_count"]

        rank_0_print("[Simulation] Setting up callbacks states")
        self._callbacks_state = [CallbackState(c, self.cur_t) for c in self._callbacks]
        rank_0_print("[Simulation] Restoring callbacks states")

        wtime = shamrock.get_wtime_sync()

        for ic, c in enumerate(self._callbacks):
            self._callbacks_state[ic].from_dict(metadata[c.name])

        # Correct the walltime to be the current walltime
        # If not done it will be the next_walltime relative to when the dump was done
        for ic, c in enumerate(self._callbacks):
            if c.walltime_interval is not None:
                self._callbacks_state[ic].next_walltime = wtime + c.walltime_interval

        # in case we checkpoint in the middle of the callback sequence
        self.trigger_and_advance_callbacks()

    def evolve_until(
        self, next_time: float, next_iter_count: int | None, next_walltime: float | None
    ):

        if next_time < self.cur_t:
            raise ValueError(f"Next callback time {next_time} is in the past")

        if next_iter_count is not None:
            if next_iter_count < self.cur_iter_count:
                raise ValueError(f"Next callback iter count {next_iter_count} is in the past")

        if next_iter_count is None:
            next_iter_count = -1
        else:
            next_iter_count = next_iter_count - self.cur_iter_count

        if next_walltime is None:
            next_walltime = -1.0

        result = self.model.evolve_until(
            next_time, niter_max=next_iter_count, max_walltime=next_walltime
        )
        self.cur_t = self.model.get_time()
        self.cur_iter_count += result.iter_count

    def trigger_and_advance_callbacks(self):
        callback_to_advance = []

        wtime = shamrock.get_wtime_sync()
        for ic, c in enumerate(self._callbacks):
            trig = self._callbacks_state[ic].should_trigger(self.cur_t, self.cur_iter_count, wtime)
            if trig:
                counter = self._callbacks_state[ic].counter
                rank_0_print("--------------------------------")
                c.func(counter)
                rank_0_print("--------------------------------")
                callback_to_advance.append(ic)

        # in case there is a long running callback this won't fuck up the walltimes
        # Also if a callback checkpoints we won't be in a partially advanced state

        for ic in callback_to_advance:
            self._callbacks_state[ic].advance(self.cur_t, self.cur_iter_count, wtime)

    def goto_run_next_callback(self):

        next_time = self.t_end
        next_iter_count = None
        next_walltime = None

        for ic, _ in enumerate(self._callbacks):
            state = self._callbacks_state[ic]

            if state.next_tsim is not None:
                next_time = min(next_time, state.next_tsim)

            if state.next_iter_count is not None:
                if next_iter_count is None:
                    next_iter_count = state.next_iter_count
                else:
                    next_iter_count = min(next_iter_count, state.next_iter_count)

            if state.next_walltime is not None:
                if next_walltime is None:
                    next_walltime = state.next_walltime
                else:
                    next_walltime = min(next_walltime, state.next_walltime)

        rank_0_print()
        rank_0_print(
            f"[Simulation] Evolve until next trigger(s) :\n"
            f"   -> t = {next_time} (current = {self.cur_t})\n"
            f"   -> iter = {next_iter_count} (current = {self.cur_iter_count})\n"
            f"   -> walltime = {next_walltime} (current = {shamrock.get_wtime_sync()})"
        )
        rank_0_print()

        self.evolve_until(next_time, next_iter_count, next_walltime)

        self.trigger_and_advance_callbacks()

    def run(self):

        if self.dump_helper is not None:
            metadata = self.dump_helper.load_last_dump_or(self.run_setup)
            if metadata is not None:
                rank_0_print("[Simulation] Restoring Simulation handle from checkpoint")
                self.restore_from_checkpoint(metadata)
        else:
            self.run_setup()

        while self.cur_t < self.t_end:
            self.goto_run_next_callback()
