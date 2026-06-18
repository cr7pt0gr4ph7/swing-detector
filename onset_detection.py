#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt


def princarg(angle: npt.ArrayLike | float) -> npt.ArrayLike | float:
    return np.mod(angle + np.pi, -2 * np.pi) + np.pi


class OnsetDetectionFunction:
    # Runtime data / state
    data_length: int
    half_length: int
    step_size: int
    dtype: npt.DTypeLike

    def prepare(
        self,
        frame_length: int,
        step_size: int,
        dtype: npt.DTypeLike,
    ):
        self.data_length = frame_length
        self.half_length = self.data_length/2 + 1
        self.step_size = step_size
        self.dtype = dtype
        self.init()

    def init(self):
        pass

    def process_frame(self, spectrum: npt.NDArray[np.complexfloating]):
        raise NotImplemented


class HighFrequencyContentODF(OnsetDetectionFunction):
    def process_frame(self, spectrum: npt.NDArray[np.complexfloating]):
        magnitudes: npt.NDArray[np.float    ing] = np.abs(spectrum)

        return np.sum(
            np.multiply(
                magnitudes,
                np.arange(1, 1 + magnitudes.shape[0])
            )
        )


class SpectralDifferenceODF(OnsetDetectionFunction):
    magnitude_history: npt.NDArray[np.floating]

    def init(self):
        self.magnitude_history = np.ndarray((self.half_length,), self.dtype)

    def process_frame(self, spectrum: npt.NDArray[np.complexfloating]):
        magnitudes: npt.NDArray[np.floating] = np.abs(spectrum)
        result = np.sum(
            np.sqrt(
                np.abs(
                    (magnitudes * magnitudes) -
                    (self.magnitude_history * self.magnitude_history)
                )
            )
        )
        self.magnitude_history = magnitudes
        return result


class PhaseDeviationODF(OnsetDetectionFunction):
    phase_history_old: npt.NDArray[np.floating]
    phase_history: npt.NDArray[np.floating]

    def init(self):
        self.phase_history = np.ndarray((self.half_length,), self.dtype)
        self.phase_history_old = np.ndarray((self.half_length,), self.dtype)

    def process_frame(self, spectrum):
        phases: npt.NDArray[np.floating] = np.angle(spectrum)

        tmp_phase = (phases - 2*self.phase_history+self.phase_history_old)
        tmp_phase_deviation = princarg(tmp_phase)

        # A previous version of this code only counted the value here
        # if the magnitude exceeded 0.1.  My impression is that
        # doesn't greatly improve the results for "loud" music (so
        # long as the peak picker is reasonably sophisticated), but
        # does significantly damage its ability to work with quieter
        # music, so I'm removing it and counting the result always.
        # Same goes for the spectral difference measure above.

        tmp_value = np.abs(tmp_phase_deviation)
        value = np.sum(tmp_value)

        self.phase_history_old = self.phase_history
        self.phase_history = phases
        return value


class BroadbandODF(OnsetDetectionFunction):
    # Runtime data / state
    magnitude_history: npt.NDArray[np.floating]

    def init(self):
        self.magnitude_history = np.ndarray((self.half_length,), self.dtype)

    def process_frame(self, spectrum):
        magnitudes: npt.NDArray[np.floating] = np.abs(spectrum)

        squared_magnitudes = magnitudes * magnitudes
        diff = 10.0 * np.log10(squared_magnitudes / self.magnitude_history)
        value = np.sum(self.magnitude_history > 0.0 and diff > self.db_rise)

        self.magnitude_history = squared_magnitudes
        return value


class ComplexSpectralDifferenceODF(OnsetDetectionFunction):
    # Parameters
    db_rise: float
    adaptive_whitening: bool
    whitening_relax_coeff: float
    whitening_floor: float

    # Runtime data / state
    phase_history_old: npt.NDArray[np.floating]
    phase_history: npt.NDArray[np.floating]
    magnitude_history: npt.NDArray[np.floating]

    def __init__(self, db_rise: float, adaptive_whitening: bool, whitening_relax_coeff: float, whitening_floor: float):
        self.db_rise = db_rise
        self.whiten = adaptive_whitening
        self.whiten_relax_coeff = whitening_relax_coeff
        self.whiten_floor = whitening_floor

        if self.whiten_relax_coeff < 0:
            self.whiten_relax_coeff = 0.9997

        if self.whiten_floor < 0:
            self.whiten_floor = 0.01

    def init(self):
        self.magnitude_history = np.ndarray((self.half_length,), self.dtype)
        self.phase_history = np.ndarray((self.half_length,), self.dtype)
        self.phase_history_old = np.ndarray((self.half_length,), self.dtype)
        self.magnitude_peaks = np.ndarray((self.half_length,), self.dtype)

    def process_frame(self, spectrum: npt.NDArray[np.complexfloating]):
        # We only get a temporal section of the track at a time
        # - namely the section corresponding to the current frame.
        magnitudes: npt.NDArray[np.floating] = np.abs(spectrum)
        phases: npt.NDArray[np.floating] = np.angle(spectrum)

        tmp_phases: npt.NDArray[np.floating] = phases - \
            2*self.phase_history+self.phase_history_old
        dev: npt.NDArray[np.floating] = princarg(tmp_phases)
        meas: npt.NDArray[np.complexfloating] = self.magnitude_history - \
            (magnitudes * np.exp(complex(0, 1) * dev))

        self.phase_history_old = self.phase_history
        self.phase_history = phases
        self.magnitude_history = magnitudes

        return np.sum(np.abs(meas))
