import numpy as np

from .ContactDiscontinuity import ContactDiscontinuity
from .Rarefaction import Rarefaction
from .Shock import Shock
from .State import State


class Wavefan:
    def __init__(self, states, waves, reversed=False):
        assert len(states) == len(waves) + 1
        if not reversed:
            self.states = states
            self.waves = waves
        else:
            self.states = Wavefan.__reverseStates(states)
            self.waves = Wavefan.__reverseWaves(waves)

        self.regionBoundaries, self.regionStates = Wavefan.__build(self.states, self.waves)

    @staticmethod
    def __reverseStates(states):
        states = list(reversed(states))
        for state in states:
            state.vx *= -1
        return states

    @staticmethod
    def __reverseWave(wave):
        if isinstance(wave, Shock):
            return Shock(wave.stateA, wave.stateB, wave.eos, -wave.sign)
        elif isinstance(wave, ContactDiscontinuity):
            return ContactDiscontinuity(-wave.speed, wave.pressure)
        elif isinstance(wave, Rarefaction):
            return Rarefaction(wave.stateA, wave.stateB, wave.eos, -wave.sign)
        else:
            raise RuntimeError("Unknown wave")

    @staticmethod
    def __reverseWaves(waves):
        return [Wavefan.__reverseWave(wave) for wave in reversed(waves)]

    @staticmethod
    def __build(states, waves):
        regionBoundaries = [-np.inf]
        regionStates = []

        regionStates.append(lambda xi: states[0])
        for idx, _ in enumerate(waves):
            Wavefan.__appendWave(regionBoundaries, regionStates, waves[idx])
            regionStates.append(lambda xi, idx=idx: states[idx + 1])

        regionBoundaries.append(np.inf)
        return regionBoundaries, regionStates

    @staticmethod
    def __appendWave(regionBoundaries, regionStates, wave):
        if isinstance(wave, Shock):
            regionBoundaries.append(wave.speed)
        elif isinstance(wave, ContactDiscontinuity):
            regionBoundaries.append(wave.speed)
        elif isinstance(wave, Rarefaction):
            regionBoundaries.append(min(wave.speedTail, wave.speedHead))
            regionStates.append(wave.computeRarefactionState)
            regionBoundaries.append(max(wave.speedTail, wave.speedHead))
        else:
            raise RuntimeError("Unknown wave")

    @staticmethod
    def __getRegionCondlist(xi, regionBoundaries):
        condlist = []
        condlist.append(xi <= regionBoundaries[0])
        for left, right in zip(regionBoundaries[:-1], regionBoundaries[1:]):
            condlist.append((left < xi) * (xi <= right))
        condlist.append(regionBoundaries[-1] < xi)
        return condlist

    def getRegionIndex(self, xi):
        return np.digitize(xi, self.regionBoundaries) - 1

    def __getState(self, xi):
        idx = self.getRegionIndex(xi)
        return self.regionStates[idx](xi)

    def getState(self, xis):
        doGet = np.vectorize(self.__getState)
        states = doGet(xis)

        try:
            statesAggregated = type(states[0])()
            for var in vars(statesAggregated):
                setattr(statesAggregated, var, np.array([getattr(state, var) for state in states]))
            return statesAggregated
        except:
            return states

    def __str__(self):
        string = "Wavefan:"
        tmp = ["", str(self.states[0])]
        for wave, state in zip(self.waves, self.states[1:]):
            tmp.append(str(wave))
            tmp.append(str(state))
        string += "\n  - ".join(tmp)
        return string

    def __repr__(self):
        return str(self)
