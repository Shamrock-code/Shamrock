import types
from dataclasses import dataclass
from math import inf
from typing import Callable

from shamrock.utils.dump import ShamrockDumpHandleHelper

# ----------------------------
# decorators
# ----------------------------


def callback(*, tsim_interval):
    """
    Decorator to mark a function as a simulation callback.

    Example:
        @callback(tsim_interval=1.0)
        def analysis(self, icallback):
            print("analysis", icallback)

    Args:
        tsim_interval: The time step of the callback.

    Returns:
        The decorated function.
    """

    def deco(func):
        func.__simulation_callback__ = {
            "func_name": func.__name__,
            "tsim_interval": tsim_interval,
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
        if name == "SimulationHandle":
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


@dataclass
class CallbackInfo:
    func: Callable
    name: str

    tsim_interval: float | None = None
    # iter_count_interval: int | None = None
    # walltime_interval: float | None = None


@dataclass
class CallbackState:
    counter: int = 0

    next_tsim: float | None = None
    # next_iter_count:int | None = None
    # next_walltime:float | None = None

    def __init__(self, info: CallbackInfo, tsim_start: float):
        self.info = info

        if self.info.tsim_interval is not None:
            self.next_tsim = tsim_start
        # if self.info.iter_count_interval is not None:
        #    self.next_iter_count = 0
        # if self.info.walltime_interval is not None:
        #    self.next_walltime = 0.0

    def advance(self, t_model: float):
        self.counter += 1

        if self.info.tsim_interval is not None:
            self.next_tsim = t_model + self.info.tsim_interval
        # if self.info.iter_count_interval is not None:
        #    self.next_iter_count += self.info.iter_count_interval
        # if self.info.walltime_interval is not None:
        #    self.next_walltime += self.info.walltime_interval

    def should_trigger(self, t_model: float) -> bool:
        trig = False
        if self.info.tsim_interval is not None:
            trig = trig or t_model >= self.next_tsim  # should i add a tolerance here ?
        # if self.info.iter_count_interval is not None:
        #    trig = trig or self.counter >= self.next_iter_count
        # if self.info.walltime_interval is not None:
        #    trig = trig or self.counter >= self.next_walltime
        return trig


class SimulationHandle(metaclass=SimulationMeta):
    """
    SimulationHandle is a base class to declare a simulation with setup & callbacks.

    A derived class must define:
    - t_end: float = <end time of the simulation>
    - a setup (any function decorated with @simulation_setup)

    And can define callbacks (any function decorated with @callback):

    < call every tsim = i * time_step >
    - @callback(time_step=1.0)
      def analysis(self, icallback):
          print("analysis")

    < call when tsim = dt_stop, niter_max is reached or walltime_step is reached >
    - @callback(time_step=dt_stop, niter_max=1000, walltime_step=30*60)
      def do_checkpoint(self, icheckpoint):
          self.dump_helper.dump(icheckpoint)

    Note that for the last one that this reset the counters until next callback.
    The trigger conditions are inclusive and reset the counters for all triggers of that callback.
    """

    t_end: float | None = None

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
            )

            self._callbacks.append(copied)

        self._callbacks_state = None

    def __post_init__(self):

        if self.t_end is None:
            raise ValueError(f"{type(self).__name__}.t_end must be defined")

        if self._declared_callbacks is None:
            raise ValueError(f"{type(self).__name__}._declared_callbacks must be defined")

        if self._setup is None:
            raise ValueError(f"{type(self).__name__}._setup must be defined")

    def run_setup(self):

        name, func = self._setup
        print(f"Running setup function: {name}")
        func(self)

        print("Setting up callbacks states")
        self._callbacks_state = [CallbackState(c, self.model.get_time()) for c in self._callbacks]

    def goto_run_next_callback(self):

        next_time = self.t_end

        for ic, _ in enumerate(self._callbacks):
            state = self._callbacks_state[ic]
            next_time = min(next_time, state.next_tsim)

        if next_time < self.model.get_time():
            raise ValueError(f"Next callback time {next_time} is in the past")

        print(f"Next triggers at t={next_time}")

        result = self.model.evolve_until(next_time)

        print(result)

        for ic, c in enumerate(self._callbacks):
            trig = self._callbacks_state[ic].should_trigger(self.model.get_time())
            if trig:
                counter = self._callbacks_state[ic].counter
                print(
                    f"Triggering callback {c.name} at t={self.model.get_time()} (counter={counter})"
                )
                c.func(counter)
                self._callbacks_state[ic].advance(self.model.get_time())

    def run(self):
        self.run_setup()

        while self.model.get_time() < self.t_end:
            self.goto_run_next_callback()
