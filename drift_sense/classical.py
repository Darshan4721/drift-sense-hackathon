"""Classical multi-scale localization for Drift-Sense."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import cv2
import numpy as np

from .config import SEARCH_SCALES
from .image_utils import apply_prefilter, gradient_magnitude, normalize_contrast


@dataclass
class Candidate:
    """Candidate center and component scores."""

    x: float
    y: float
    width: int
    height: int
    scale: float
    rotation_degrees: float
    intensity_score: float
    edge_score: float
    structure_score: float
    periodicity_score: float
    distance_score: float = 0.0
    combined_score: float = 0.0
    classical_score: float = 0.0
    verifier_score: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _correlation_map(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Compute normalized cross-correlation after robust contrast normalization."""

    return cv2.matchTemplate(
        normalize_contrast(search).astype(np.float32),
        normalize_contrast(template).astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )


def _top_local_maxima(score_map: np.ndarray, k: int, min_distance: int) -> list[tuple[int, int, float]]:
    """Return spatially separated top-left maxima from a score map."""

    work = score_map.copy()
    candidates: list[tuple[int, int, float]] = []
    for _ in range(max(1, k)):
        _, maximum, _, location = cv2.minMaxLoc(work)
        x, y = int(location[0]), int(location[1])
        if maximum <= -1.0:
            break
        candidates.append((x, y, float(maximum)))
        x0, x1 = max(0, x - min_distance), min(work.shape[1], x + min_distance + 1)
        y0, y1 = max(0, y - min_distance), min(work.shape[0], y + min_distance + 1)
        work[y0:y1, x0:x1] = -1.0
    return candidates


def _local_feature_score(search: np.ndarray, template: np.ndarray, x: int, y: int) -> tuple[float, float, float, float]:
    """Score intensity, edge, structural smoothness, and periodic consistency."""

    height, width = template.shape
    patch = search[y : y + height, x : x + width]
    if patch.shape != template.shape:
        return -1.0, -1.0, -1.0, -1.0
    template_n = normalize_contrast(template)
    patch_n = normalize_contrast(patch)
    edge_template = gradient_magnitude(template_n)
    edge_patch = gradient_magnitude(patch_n)
    intensity = float(cv2.matchTemplate(patch_n, template_n, cv2.TM_CCOEFF_NORMED)[0, 0])
    edge = float(cv2.matchTemplate(edge_patch, edge_template, cv2.TM_CCOEFF_NORMED)[0, 0])
    template_blur = cv2.GaussianBlur(template_n, (0, 0), 1.2)
    patch_blur = cv2.GaussianBlur(patch_n, (0, 0), 1.2)
    structure = float(cv2.matchTemplate(patch_blur, template_blur, cv2.TM_CCOEFF_NORMED)[0, 0])
    # Projection profiles capture repeating line spacing without assuming an exact
    # DRAM period. The score compares normalized profile derivatives.
    t_profile = np.concatenate([edge_template.mean(axis=0), edge_template.mean(axis=1)])
    p_profile = np.concatenate([edge_patch.mean(axis=0), edge_patch.mean(axis=1)])
    t_profile = (t_profile - t_profile.mean()) / (t_profile.std() + 1e-6)
    p_profile = (p_profile - p_profile.mean()) / (p_profile.std() + 1e-6)
    periodicity = float(np.mean(t_profile * p_profile) / (np.sqrt(np.mean(t_profile**2) * np.mean(p_profile**2)) + 1e-6))
    return intensity, edge, structure, float(np.clip(periodicity, -1.0, 1.0))


def generate_candidates(
    reference: np.ndarray,
    search: np.ndarray,
    scales: Iterable[float] = SEARCH_SCALES,
    top_k_per_scale: int = 8,
    rotations: Iterable[float] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    prefilter: str = "none",
) -> list[Candidate]:
    """Generate candidate centers over the known approximately 0.1 scale range."""

    ref_norm = normalize_contrast(reference)
    search_norm = normalize_contrast(search)
    if prefilter != "none":
        ref_corr = apply_prefilter(ref_norm, prefilter)
        search_corr = apply_prefilter(search_norm, prefilter)
    else:
        ref_corr = ref_norm
        search_corr = search_norm
    all_candidates: list[Candidate] = []
    for scale in scales:
        width = max(24, int(round(reference.shape[1] * float(scale))))
        height = max(24, int(round(reference.shape[0] * float(scale))))
        if width >= search.shape[1] or height >= search.shape[0]:
            continue
        base_template_corr = cv2.resize(ref_corr, (width, height), interpolation=cv2.INTER_AREA)
        base_template_orig = cv2.resize(ref_norm, (width, height), interpolation=cv2.INTER_AREA) if prefilter != "none" else base_template_corr
        for rotation in rotations:
            if abs(float(rotation)) > 1e-6:
                matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(rotation), 1.0)
                template_corr = cv2.warpAffine(base_template_corr, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
                template_orig = cv2.warpAffine(base_template_orig, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101) if prefilter != "none" else template_corr
            else:
                template_corr = base_template_corr
                template_orig = base_template_orig
            score_map = _correlation_map(search_corr, template_corr)
            for x, y, _ in _top_local_maxima(score_map, top_k_per_scale, max(8, min(width, height) // 3)):
                intensity, edge, structure, periodicity = _local_feature_score(search_norm, template_orig, x, y)
                all_candidates.append(
                    Candidate(
                        x=x + width / 2.0,
                        y=y + height / 2.0,
                        width=width,
                        height=height,
                        scale=float(scale),
                        rotation_degrees=float(rotation),
                        intensity_score=intensity,
                        edge_score=edge,
                        structure_score=structure,
                        periodicity_score=periodicity,
                    )
                )
    return all_candidates


def rerank_candidates(
    candidates: list[Candidate],
    search_shape: tuple[int, int],
    verifier_scores: dict[int, float] | None = None,
) -> list[Candidate]:
    """Combine appearance, structure, periodicity, center prior, and verifier scores."""

    if not candidates:
        return []
    height, width = search_shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    diagonal = float(np.hypot(width, height))
    for index, candidate in enumerate(candidates):
        distance = float(np.hypot(candidate.x - center_x, candidate.y - center_y) / diagonal)
        candidate.distance_score = float(1.0 - np.clip(distance, 0.0, 1.0))
        base = (
            0.55 * candidate.intensity_score
            + 0.18 * candidate.edge_score
            + 0.20 * candidate.structure_score
            + 0.04 * candidate.periodicity_score
            + 0.03 * candidate.distance_score
        )
        candidate.classical_score = base
        if verifier_scores and index in verifier_scores:
            candidate.verifier_score = float(verifier_scores[index])
            candidate.combined_score = 0.78 * base + 0.22 * candidate.verifier_score
        else:
            candidate.combined_score = base
    # When scores are nearly indistinguishable, the specification prefers the
    # candidate closest to the search-image center. A small absolute margin is
    # more stable than a relative margin for normalized correlations.
    best_score = max(candidate.combined_score for candidate in candidates)
    near_ties = [candidate for candidate in candidates if best_score - candidate.combined_score <= 0.003]
    near_tie_ids = {id(candidate) for candidate in near_ties}
    near_ties.sort(key=lambda item: np.hypot(item.x - center_x, item.y - center_y))
    remaining = [candidate for candidate in candidates if id(candidate) not in near_tie_ids]
    remaining.sort(key=lambda item: (-item.combined_score, np.hypot(item.x - center_x, item.y - center_y)))
    return near_ties + remaining


def localize_classical(
    reference: np.ndarray,
    search: np.ndarray,
    scales: Iterable[float] = SEARCH_SCALES,
    top_k_per_scale: int = 8,
) -> tuple[Candidate, list[Candidate]]:
    """Return the best candidate and the full ranked candidate list."""

    candidates = generate_candidates(reference, search, scales, top_k_per_scale)
    ranked = rerank_candidates(candidates, search.shape)
    if not ranked:
        raise RuntimeError("No valid localization candidates were generated")
    return ranked[0], ranked
