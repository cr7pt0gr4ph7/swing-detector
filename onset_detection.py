#!/usr/bin/env python3

import librosa
import numpy as np
import numpy.typing as npt


def princarg(ang: float) -> float:
    return np.mod(ang + np.pi, -2 * np.pi) + np.pi


class ComplexSDDetectionFunction:
    def __init__(self):
        self.phaseHistoryOld: list[float] = []
        self.phaseHistory: list[float] = []
        self.magnitudeHistory: list[float] = []

    def complex_sd(self, length: int, spectrum: npt.NDArray[np.complexfloating]):
        # We only get temporal section of the section - namely
        # the section corresponding to the current frame.
        magnitudes: npt.NDArray[np.floating] = np.abs(spectrum)
        phases: npt.NDArray[np.floating] = np.angle(spectrum)

        val: float = 0.0
        j: complex = complex(0.0, 1.0)

        for freq_index in range(0, length):
            tmpPhase = (
                phases[freq_index] - 2*self.phaseHistory[freq_index]+self.phaseHistoryOld[freq_index])
            dev = princarg(tmpPhase)

            meas: complex = self.magnitudeHistory[freq_index] - \
                (magnitudes[freq_index] * np.exp(j * dev))
            val += np.abs(meas)

            self.phaseHistoryOld[freq_index] = self.phaseHistory[freq_index]
            self.phaseHistory[freq_index] = phases[freq_index]
            self.magnitudeHistory[freq_index] = magnitudes[freq_index]

        return val
