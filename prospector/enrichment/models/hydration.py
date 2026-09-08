"""Hydration class: how aqueously altered a body's surface is.

Water is the reason most near-Earth asteroids are worth visiting at all: as propellant, as
consumables, and as the marker of the carbonaceous material that carries everything else of
interest. It cannot be observed directly, but aqueous alteration leaves a spectral record, and
meteorites whose water content was measured in a laboratory tie that record to a number.

This classifier is trained on those meteorite spectra. Its classes are the petrologic types that
describe how much liquid water processed the parent body:

- ``H12`` -- types 1-2, the most heavily altered, water-rich
- ``H3``  -- type 3, minimally processed
- ``H4``  -- type 4, thermally metamorphosed, water largely driven off
- ``H5``  -- type 5, strongly metamorphosed and dry

The features are the body's albedo, the depth of its 0.7 um hydration band
(:func:`~prospector.enrichment.models.reflectance.band_depth_07um`), and the first four
principal-component scores of its spectrum in the Mahlke et al. (2022) taxonomy. Those scores
require the ``classy`` package, which is an optional dependency: without it this module reports an
unknown class and enrichment continues without the hydration column.

The classifier is a small feed-forward network, stored as plain weight arrays and evaluated here in
a few lines of numpy. It is not a pickled estimator object, because a pickle executes code on load,
silently changes meaning between library versions, and would drag a machine-learning framework into
the dependency set to run three matrix multiplications.

The weights were fitted to 630 lab spectra of meteorites from the RELAB collection at Brown
University (https://sites.brown.edu/relab/), which NASA distributes through the Planetary Data
System. Please credit RELAB if you publish anything built on this. The spectra are not included
here, just the weights.

It gets about 94% of the held-back test data right, but that number is generous: 429 of the 630
spectra are H5, so guessing dry is usually correct. It is dependable at the dry and water-rich ends
and much weaker on the two middle classes, which had far fewer examples. H3 came down to four test
samples.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from prospector.enrichment.models.reflectance import Color, band_depth_07um, colors_to_spectrum

# Class labels in the order the model's probability vector is reported.
HYDRATION_CLASSES: tuple[str, ...] = ("H12", "H3", "H4", "H5")

# Feature vector layout, in the order the network was trained on. Named so a caller assembling
# features cannot silently transpose two of them.
FEATURE_ORDER: tuple[str, ...] = ("albedo", "band_depth_07um", "z0", "z1", "z2", "z3")

# The trained weights, kept beside this module: they are a model, not configuration, and swapping
# them changes what the tool believes rather than how it is set up.
MODEL_PATH = Path(__file__).with_name("hydration_model.npz")

# Taxonomy the principal-component scores are defined in. The model's features are that taxonomy's
# scores specifically, so this is not a free choice.
TAXONOMY = "mahlke"

_model: HydrationModel | None = None


class HydrationUnavailable(RuntimeError):
    """The classifier cannot run here: its optional dependencies or model file are missing."""


class HydrationModel:
    """A trained feed-forward classifier: rectified hidden layers, softmax over the classes."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]], class_labels: Sequence):
        self.layers = layers
        # The training labels are the bare petrologic types (3, 4, 5, 12); the "H" prefix is this
        # module's naming. Mapping through the stored labels means a retrained model cannot
        # silently permute which probability belongs to which class.
        self.class_names = [f"H{int(label)}" for label in class_labels]

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Class probabilities for one feature vector, in :attr:`class_names` order."""
        activations = np.asarray(features, dtype=float).reshape(-1)
        *hidden, (out_w, out_b) = self.layers
        for weights, bias in hidden:
            activations = np.maximum(activations @ weights + bias, 0.0)
        logits = activations @ out_w + out_b
        exponentials = np.exp(logits - np.max(logits))    # shifted for numerical stability
        return exponentials / exponentials.sum()


def load_model(path: Path | None = None) -> HydrationModel:
    """The trained classifier, loaded once and reused.

    Raises :class:`HydrationUnavailable` instead of an I/O error, so a caller can treat "no
    hydration class" uniformly whether the cause is a missing model file, a missing optional
    package, or a body with no spectrum.
    """
    global _model
    if _model is not None and path is None:
        return _model

    target = Path(path) if path is not None else MODEL_PATH
    if not target.is_file():
        raise HydrationUnavailable(f"hydration model not found: {target}")
    try:
        with np.load(target) as stored:
            depth = sum(1 for key in stored.files if key.startswith("w"))
            layers = [(stored[f"w{i}"], stored[f"b{i}"]) for i in range(depth)]
            model = HydrationModel(layers, stored["classes"])
    except Exception as exc:
        raise HydrationUnavailable(f"hydration model would not load: {exc}") from exc

    if path is None:
        _model = model
    return model


def _spectrum_scores(wave_um: Sequence[float], reflectance: Sequence[float],
                     reflectance_err: Sequence[float] | None = None):
    """The first four Mahlke principal-component scores for a spectrum, via ``classy``.

    The albedo is blanked before classification because it is a model feature in its own right, and
    letting the taxonomy consume it too would feed the same measurement in twice.
    """
    try:
        import classy
    except ImportError as exc:
        raise HydrationUnavailable("classy is not installed") from exc

    spectrum = classy.Spectrum(wave=list(wave_um), refl=list(reflectance),
                               refl_err=list(reflectance_err) if reflectance_err else None)
    spectrum.pV = np.nan
    spectrum.classify(taxonomy=TAXONOMY)
    scores = getattr(spectrum, "scores_mahlke", None)
    if scores is None or len(scores) < 4:
        raise HydrationUnavailable("spectrum produced no principal-component scores")
    return np.asarray(scores[:4], dtype=float), spectrum


def published_spectrum(target: str):
    """The first usable published spectrum for ``target``, as ``(wave_um, reflectance, albedo)``.

    ``classy`` aggregates the public spectral surveys; the albedo it carries is read off before
    classification because it is one of the classifier's features.
    """
    try:
        import classy
    except ImportError as exc:
        raise HydrationUnavailable("classy is not installed") from exc

    spectra = classy.Spectra(target)
    if not spectra:
        raise HydrationUnavailable(f"no published spectra for '{target}'")

    for spectrum in spectra:
        try:
            albedo = getattr(spectrum, "pV", None)
            albedo = float(albedo) if albedo is not None and np.isfinite(float(albedo)) else None
        except (TypeError, ValueError):
            albedo = None
        wave = np.asarray(getattr(spectrum, "wave", []), dtype=float)
        refl = np.asarray(getattr(spectrum, "refl", []), dtype=float)
        if wave.size >= 2 and refl.size == wave.size:
            return wave, refl, albedo
    raise HydrationUnavailable(f"no usable published spectrum for '{target}'")


def classify_spectrum(wave_um: Sequence[float], reflectance: Sequence[float], albedo: float,
                      reflectance_err: Sequence[float] | None = None,
                      model: HydrationModel | None = None) -> dict[str, float]:
    """Hydration-class probabilities for one spectrum, keyed by :data:`HYDRATION_CLASSES`."""
    scores, resampled = _spectrum_scores(wave_um, reflectance, reflectance_err)
    depth = band_depth_07um(resampled.wave, resampled.refl)
    features = np.array([float(albedo), float(depth), *scores])

    classifier = model if model is not None else load_model()
    by_name = dict(zip(classifier.class_names, classifier.probabilities(features)))
    return {name: float(by_name.get(name, float("nan"))) for name in HYDRATION_CLASSES}


def classify(target: str, *, albedo: float, colors: Sequence[Color] | None = None,
             spectrum: tuple[Sequence[float], Sequence[float], Sequence[float] | None] | None = None,
             model=None) -> tuple[str, dict[str, float]]:
    """The most probable hydration class for ``target``, with the full probability vector.

    Spectra are tried in order of quality: an explicitly supplied one (a survey measurement the
    caller already has), then a published spectrum found through ``classy``, then a spectrum
    reconstructed from broadband colours. Raises :class:`HydrationUnavailable` when none of the
    three yields a classifiable spectrum.
    """
    attempts: list[tuple[Sequence[float], Sequence[float], Sequence[float] | None, float]] = []
    if spectrum is not None:
        wave, refl, err = spectrum
        attempts.append((wave, refl, err, albedo))
    try:
        wave, refl, published_albedo = published_spectrum(target)
        attempts.append((wave, refl, None, published_albedo if published_albedo else albedo))
    except HydrationUnavailable:
        pass
    if colors:
        try:
            wave, refl = colors_to_spectrum(colors)
            attempts.append((wave, refl, None, albedo))
        except ValueError:
            pass

    last_error: Exception | None = None
    for wave, refl, err, feature_albedo in attempts:
        try:
            probabilities = classify_spectrum(wave, refl, feature_albedo, err, model=model)
        except Exception as exc:
            last_error = exc
            continue
        best = max(probabilities, key=lambda name: probabilities[name])
        return best, probabilities

    if last_error is not None:
        raise HydrationUnavailable(f"could not classify '{target}': {last_error}") from last_error
    raise HydrationUnavailable(f"no spectrum available for '{target}'")
