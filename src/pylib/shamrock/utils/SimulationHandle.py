import types
from dataclasses import dataclass
from typing import Callable

from shamrock.utils.dump import ShamrockDumpHandleHelper

# ----------------------------
# decorators
# ----------------------------


def callback(*, time_step):
    """
    Decorator to mark a function as a simulation callback.

    Example:
        @callback(time_step=1.0)
        def analysis(self, icallback):
            print("analysis", icallback)

    Args:
        time_step: The time step of the callback.

    Returns:
        The decorated function.
    """

    def deco(func):
        func.__simulation_callback__ = {
            "time_step": time_step,
            "func_name": func.__name__,
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

    time_step: float | None = None

    def compute_next_time(self, t_model: float) -> float:

        if self.time_step is None:
            raise ValueError(f"{type(self).__name__}.time_step must be defined")

        t_model_modf = t_model % self.time_step

        if t_model_modf == 0:
            return t_model + self.time_step

        return t_model + self.time_step - t_model_modf


class SimulationHandle(metaclass=SimulationMeta):
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
                time_step=info["time_step"],
            )

            self._callbacks.append(copied)

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

    def goto_run_next_callback(self):
        cur_t = self.model.get_time()

        next_time = self.t_end
        callbacks_to_execute = []

        for c in self._callbacks:
            next_time_callback = c.compute_next_time(cur_t)
            if next_time_callback < next_time:
                next_time = next_time_callback
                callbacks_to_execute.clear()
                callbacks_to_execute.append(c)
            elif next_time_callback == next_time:
                callbacks_to_execute.append(c)

        return next_time, callbacks_to_execute

    def run(self):
        self.run_setup()

        while self.model.get_time() < self.t_end:
            self.goto_run_next_callback()
