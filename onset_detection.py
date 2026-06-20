#!/usr/bin/env python3

from typing import Optional

import librosa
import numpy as np
import numpy.typing as npt


def princarg(angle: npt.ArrayLike | float) -> npt.ArrayLike | float:
    return np.mod(angle + np.pi, -2 * np.pi) + np.pi


class OnsetDetectionFunction:
    # Runtime data / state
    frame_length: int
    half_length: int
    step_size: int
    dtype: npt.DTypeLike

    def run(
        self,
        y: npt.NDArray[np.floating],
        sr: Optional[float] = None,
        block_size: int = None,  # corresponds to n_fft in librosa?
        step_size: int = None,  # corresponds to hop_length in librosa
    ):
        if step_size is None:
            if sr is None:
                raise ValueError(
                    "At least one of sr and step_size must be specified")

            preferred_step_seconds = 0.01161
            preferred_step_size = max(
                1, int(sr * preferred_step_seconds + 0.0001))
            step_size = preferred_step_size

        if block_size is None:
            preferred_step_size = step_size
            preferred_block_size = 2 * preferred_step_size
            block_size = preferred_block_size

        S = librosa.stft(y=y, n_fft=block_size, hop_length=step_size)

        return self.run_on_spectogram(S, block_size, step_size, y.dtype)

    def run_on_spectogram(
        self,
        spectogram: npt.NDArray[np.complexfloating],
        block_size: int,
        step_size: int,
        dtype: npt.DTypeLike,
    ):
        self.prepare(block_size, step_size, dtype)

        spectogram_t = np.transpose(spectogram)
        result = np.ndarray((spectogram_t.shape[0],), self.dtype)

        for i in range(0, spectogram_t.shape[0]):
            # represents block_size samples, picked at every step_size
            spectrum = spectogram_t[i]
            result[i] = self.process_frame(spectrum)

        return result

    def prepare(
        self,
        block_size: int,
        step_size: int,
        dtype: npt.DTypeLike,
    ):
        self.frame_length = int(block_size)
        self.half_length = int(self.frame_length/2 + 1)
        self.step_size = int(step_size)
        self.dtype = dtype
        self.init()

    def init(self):
        pass

    def process_frame(self, spectrum: npt.NDArray[np.complexfloating]):
        raise NotImplemented


class HighFrequencyContentODF(OnsetDetectionFunction):
    def process_frame(self, spectrum: npt.NDArray[np.complexfloating]):
        magnitudes: npt.NDArray[np.floating] = np.abs(spectrum)

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

    def __init__(
        self,
        sensitivity: float,
        adaptive_whitening: bool,
        whitening_relax_coeff: float = -1,
        whitening_floor: float = -1,
    ):
        self.db_rise = 6.0 - sensitivity / 16.6667
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
