# Drift-Sense citations and design rationale

This document distinguishes **engineering approximations** from claims about real semiconductor imaging. The generator is intentionally synthetic and does not claim to reproduce a particular foundry process or SEM instrument.

## Structural references

1. **DRAM array organization.** The DRAM generator uses orthogonal horizontal word-lines, vertical bit-lines, and contact-like features at intersections. This is a simplified visual abstraction of array organization; it is not a process-accurate layout. A Virginia Tech DRAM PUF paper describes DRAM cells grouped in arrays with rows connected to word-lines and columns connected to bit-lines: [Xiong et al., “DRAM PUFs in Commodity Devices,” IEEE Design & Test, 2021](https://bearhw.ece.vt.edu/content/dam/bearhw_ece_vt_edu/publications/2021_IEEEDT_DRAM.pdf).

2. **FinFET option.** The optional FinFET generator draws repeated vertical fins and horizontal gate bars. It is motivated by the review’s discussion of fin dimensions, double/trigate structures, and gates controlling a fin-shaped channel: [Khan et al., “Comprehensive Review of FinFET Technology: History, Structure, Challenges, Innovations, and Emerging Sensing Applications,” *Micromachines*, 2024](https://doi.org/10.3390/mi15101187).

## Image formation and degradation choices

| Implemented component | Why it is included | Reference | Parameter range in this repository | Reasonableness note |
|---|---|---|---|---|
| Independent Gaussian sensor noise | Represents additive readout and electronics variation; the search view is intentionally noisier. | [Ong and Sim, “Single image signal-to-noise ratio estimation techniques for scanning electron microscope: A review,” *IEEE Access*, 2024](https://ieeexplore.ieee.org/abstract/document/10720051/) | Reference sigma 2/255; search sigma 5/255 | Small relative to the normalized [0, 1] intensity range, yet visible after downsampling. |
| Poisson-like shot noise | Represents signal-dependent counting variation using independent random draws. | [Kockentiedt et al., “Poisson shot noise parameter estimation from a single scanning electron microscopy image,” SPIE, 2013](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/8655/1/Poisson-shot-noise-parameter-estimation-from-a-single-scanning-electron/10.1117/12.2008374.full) | Reference strength 20; search strength 10 | The implementation is a lightweight simulation, not a calibrated SEM noise model. |
| Gaussian blur | Models finite resolution, focus, and resampling blur. | [OpenCV Sobel Derivatives tutorial](https://docs.opencv.org/4.13.0/d2/d2c/tutorial_sobel_derivatives.html) | Reference sigma 0.6; search sigma 1.2 | Search blur is stronger because the lower-magnification image is additionally downsampled. |
| Low-frequency illumination variation | Avoids an unrealistically stationary background and tests contrast normalization. | [Salwig et al., “Zero-shot denoising of microscopy images recorded at high noise levels,” 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11230634/) | Standard deviation 0.08 on a coarse field | The field is deliberately smooth so it changes illumination rather than geometry. |
| Contrast variation | Models gain and local imaging-condition differences. | [Thomas and Gehrig, “Multi-template matching: a versatile tool for object-localization in microscopy images,” *BMC Bioinformatics*, 2020](https://doi.org/10.1186/s12859-020-3363-7) | Multiplicative contrast 0.90–1.10 | Mild gain changes preserve the underlying pattern while challenging raw intensity matching. |
| Rotation, scaling, translation | Models small pose and calibration differences between magnifications. | [Thomas and Gehrig, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7003318/) | Rotation ±1.5°, scale 0.985–1.015, translation ±3 pixels | These are small perturbations around the known 10x relationship, not arbitrary geometric distortion. |
| Edge brightening | Adds a configurable gradient-derived term to test edge-aware scoring. | [OpenCV Sobel Derivatives tutorial](https://docs.opencv.org/4.13.0/d2/d2c/tutorial_sobel_derivatives.html) | Alpha 0.00–0.08 | Sobel magnitude is normalized per image; alpha is intentionally weak. |
| Impulse noise | Tests robustness to isolated defective pixels. | [Ong and Sim, 2024](https://ieeexplore.ieee.org/abstract/document/10720051/) | Probability 0.0004 | Rare impulses avoid turning the synthetic scene into a salt-and-pepper benchmark. |

The reference and search views use separate random generators, `seed + 1` and `seed + 2`, so the same noise field is never reused.

## Localization references

3. **Normalized cross-correlation and candidate maxima.** Stage 1 uses OpenCV `matchTemplate` with normalized cross-correlation. OpenCV documents template matching as sliding a template over a larger image, producing a score map whose extrema indicate candidate locations: [OpenCV, “Template Matching”](https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html).

4. **Multiple templates and transformed views.** The implementation searches a small set of scales and rotations, keeps separated local maxima, and reranks candidates. This follows the practical design of multi-template microscopy localization, including transformed templates and non-maxima suppression: [Thomas and Gehrig, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7003318/).

5. **Gradient and edge features.** Sobel x/y derivatives are combined into gradient magnitude, matching the standard formulation documented by OpenCV: [OpenCV, “Sobel Derivatives”](https://docs.opencv.org/4.13.0/d2/d2c/tutorial_sobel_derivatives.html).

6. **Resampling.** The generator uses OpenCV area interpolation for native search-to-image downsampling and cubic interpolation for reference upsampling. OpenCV’s geometric-transformation documentation provides the API family and interpolation behavior: [OpenCV, “Geometric Image Transformations”](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).

7. **Learned verification.** The optional `TinyVerifier` is a compact two-channel CNN trained on reference/candidate pairs with binary cross-entropy. It is deliberately a verifier rather than a coordinate-only regressor, so the learned component receives both the reference image and a search crop. If the checkpoint is absent or invalid, inference falls back to the deterministic classical path.

## Reproducibility and attribution

All code in this repository is original implementation for the challenge. External sources are used to justify design choices, not to supply executable code or hidden data. The repository does not download data or model weights during inference.
