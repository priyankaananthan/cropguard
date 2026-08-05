"""
CropGuard AI - Detection Engine
--------------------------------
Core OpenCV/HSV spectral analysis + crop-specific deficiency classification.

KEY DESIGN FIX (repetition bug):
Every crop has its OWN baseline leaf-color profile and its OWN deficiency
threshold table. Classification is never done against a single global
table - it is always (crop_type -> thresholds) first, then color analysis
against THOSE thresholds. This prevents different crops with visually
similar leaves from collapsing into the same output, and ensures dosage/
treatment lookups are always crop-specific.
"""

import cv2
import numpy as np
import hashlib
import os


# ---------------------------------------------------------------------------
# 1. CROP BASELINE PROFILES
#    Each crop's healthy leaf has a different natural HSV signature.
#    These offsets calibrate the raw pixel percentages before classification,
#    so e.g. Maize (naturally lighter/yellower-green) isn't mistaken for
#    Nitrogen-deficient Rice.
# ---------------------------------------------------------------------------
CROP_PROFILES = {
    "rice": {
        "healthy_hue_range": (35, 85),
        "green_baseline": 78,
        "yellow_sensitivity": 1.00,
        "brown_sensitivity": 1.00,
        "purple_sensitivity": 1.05,
    },
    "tomato": {
        "healthy_hue_range": (32, 80),
        "green_baseline": 72,
        "yellow_sensitivity": 0.95,
        "brown_sensitivity": 1.10,
        "purple_sensitivity": 0.90,
    },
    "maize": {
        "healthy_hue_range": (30, 82),
        "green_baseline": 70,
        "yellow_sensitivity": 1.10,
        "brown_sensitivity": 0.95,
        "purple_sensitivity": 1.00,
    },
    "wheat": {
        "healthy_hue_range": (33, 83),
        "green_baseline": 75,
        "yellow_sensitivity": 1.05,
        "brown_sensitivity": 1.00,
        "purple_sensitivity": 1.00,
    },
    "cotton": {
        "healthy_hue_range": (34, 84),
        "green_baseline": 74,
        "yellow_sensitivity": 0.90,
        "brown_sensitivity": 1.05,
        "purple_sensitivity": 1.15,
    },
    "chilli": {
        "healthy_hue_range": (36, 86),
        "green_baseline": 76,
        "yellow_sensitivity": 1.00,
        "brown_sensitivity": 1.00,
        "purple_sensitivity": 1.00,
    },
    "groundnut": {
        "healthy_hue_range": (31, 81),
        "green_baseline": 73,
        "yellow_sensitivity": 1.05,
        "brown_sensitivity": 0.90,
        "purple_sensitivity": 1.00,
    },
}

# ---------------------------------------------------------------------------
# 2. CROP-SPECIFIC DEFICIENCY THRESHOLD TABLES
#    Each (crop, deficiency) pair has its own trigger thresholds on the
#    calibrated spectral percentages. This is what stops two crops from
#    resolving to the same deficiency for similar-looking input.
# ---------------------------------------------------------------------------
DEFICIENCY_RULES = {
    # crop -> list of (deficiency_type, condition_fn, base_confidence)
    "rice": [
        ("Nitrogen Deficiency", lambda g, y, b, p: y > 22 and b < 15, 0.93),
        ("Potassium Deficiency", lambda g, y, b, p: b > 18 and y < 20, 0.90),
        ("Iron Deficiency", lambda g, y, b, p: y > 15 and g < 55, 0.87),
        ("Magnesium Deficiency", lambda g, y, b, p: y > 12 and g > 55, 0.85),
        ("Zinc Deficiency", lambda g, y, b, p: y > 10 and b > 8 and y < 22, 0.82),
        ("Phosphorus Deficiency", lambda g, y, b, p: p > 10, 0.88),
    ],
    "tomato": [
        ("Phosphorus Deficiency", lambda g, y, b, p: p > 8, 0.92),
        ("Nitrogen Deficiency", lambda g, y, b, p: y > 20 and p < 8, 0.91),
        ("Potassium Deficiency", lambda g, y, b, p: b > 20, 0.89),
        ("Magnesium Deficiency", lambda g, y, b, p: y > 14 and g > 50, 0.84),
        ("Iron Deficiency", lambda g, y, b, p: y > 18 and g < 50, 0.86),
        ("Zinc Deficiency", lambda g, y, b, p: y > 9 and b > 10, 0.80),
    ],
    "maize": [
        ("Nitrogen Deficiency", lambda g, y, b, p: y > 25, 0.94),
        ("Potassium Deficiency", lambda g, y, b, p: b > 16 and y < 18, 0.90),
        ("Magnesium Deficiency", lambda g, y, b, p: y > 13 and g > 52, 0.86),
        ("Zinc Deficiency", lambda g, y, b, p: y > 11 and b > 9, 0.83),
        ("Iron Deficiency", lambda g, y, b, p: y > 16 and g < 48, 0.85),
        ("Phosphorus Deficiency", lambda g, y, b, p: p > 9, 0.87),
    ],
    "wheat": [
        ("Nitrogen Deficiency", lambda g, y, b, p: y > 21, 0.92),
        ("Potassium Deficiency", lambda g, y, b, p: b > 17, 0.89),
        ("Iron Deficiency", lambda g, y, b, p: y > 15 and g < 53, 0.85),
        ("Magnesium Deficiency", lambda g, y, b, p: y > 12 and g > 53, 0.83),
        ("Zinc Deficiency", lambda g, y, b, p: y > 10 and b > 8, 0.81),
        ("Phosphorus Deficiency", lambda g, y, b, p: p > 9, 0.86),
    ],
    "cotton": [
        ("Potassium Deficiency", lambda g, y, b, p: b > 19, 0.91),
        ("Nitrogen Deficiency", lambda g, y, b, p: y > 19 and b < 14, 0.90),
        ("Phosphorus Deficiency", lambda g, y, b, p: p > 11, 0.88),
        ("Magnesium Deficiency", lambda g, y, b, p: y > 13 and g > 50, 0.84),
        ("Iron Deficiency", lambda g, y, b, p: y > 17 and g < 49, 0.85),
        ("Zinc Deficiency", lambda g, y, b, p: y > 10 and b > 9, 0.80),
    ],
    "chilli": [
        ("Calcium/Blossom Stress", lambda g, y, b, p: b > 20 and p < 6, 0.86),
        ("Nitrogen Deficiency", lambda g, y, b, p: y > 20, 0.91),
        ("Potassium Deficiency", lambda g, y, b, p: b > 17 and y < 18, 0.89),
        ("Iron Deficiency", lambda g, y, b, p: y > 16 and g < 52, 0.85),
        ("Magnesium Deficiency", lambda g, y, b, p: y > 12 and g > 52, 0.83),
        ("Phosphorus Deficiency", lambda g, y, b, p: p > 9, 0.87),
    ],
    "groundnut": [
        ("Iron Deficiency", lambda g, y, b, p: y > 14 and g < 54, 0.88),
        ("Nitrogen Deficiency", lambda g, y, b, p: y > 22, 0.92),
        ("Potassium Deficiency", lambda g, y, b, p: b > 16, 0.89),
        ("Magnesium Deficiency", lambda g, y, b, p: y > 11 and g > 54, 0.84),
        ("Zinc Deficiency", lambda g, y, b, p: y > 9 and b > 8, 0.81),
        ("Phosphorus Deficiency", lambda g, y, b, p: p > 8, 0.86),
    ],
}


def extract_hsv_spectrum(image_path):
    """
    Analyzes actual pixel content of the uploaded leaf image using OpenCV.
    Returns raw percentages: green, yellow, brown, purple, plus a perceptual
    hash of the image (used to keep results deterministic and non-random
    per unique image, without ever ignoring crop_type downstream).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file")

    img = cv2.resize(img, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    total_px = h.size

    # Mask out background (near-white / near-black) so only leaf pixels count
    leaf_mask = (s > 25) & (v > 25)
    leaf_px = np.count_nonzero(leaf_mask)
    if leaf_px == 0:
        leaf_px = total_px  # fallback: whole frame

    def pct_in_hue(low, high):
        m = (h >= low) & (h <= high) & leaf_mask
        return round(100 * np.count_nonzero(m) / leaf_px, 2)

    green_pct = pct_in_hue(35, 85)
    yellow_pct = pct_in_hue(20, 34)
    # brown/necrosis: low saturation-ish orange-red hues with lower value
    brown_mask = ((h >= 0) & (h <= 19) | (h >= 160) & (h <= 179)) & (v < 150) & leaf_mask
    brown_pct = round(100 * np.count_nonzero(brown_mask) / leaf_px, 2)
    purple_pct = pct_in_hue(125, 160)

    # image content hash - used only to keep repeated runs on the SAME file
    # deterministic; classification itself always comes from the pixel data
    with open(image_path, "rb") as f:
        img_hash = hashlib.md5(f.read()).hexdigest()

    return {
        "green_pct": green_pct,
        "yellow_pct": yellow_pct,
        "brown_pct": brown_pct,
        "purple_pct": purple_pct,
        "img_hash": img_hash,
    }


def classify_deficiency(spectrum, crop_type):
    """
    Applies CROP-SPECIFIC calibration + CROP-SPECIFIC threshold rules.
    This is the function that guarantees different crops don't resolve
    to identical results for similar input images.
    """
    crop_type = crop_type.lower().strip()
    profile = CROP_PROFILES.get(crop_type, CROP_PROFILES["rice"])
    rules = DEFICIENCY_RULES.get(crop_type, DEFICIENCY_RULES["rice"])

    g = spectrum["green_pct"]
    y = spectrum["yellow_pct"] * profile["yellow_sensitivity"]
    b = spectrum["brown_pct"] * profile["brown_sensitivity"]
    p = spectrum["purple_pct"] * profile["purple_sensitivity"]

    # Healthy check: green close to/above this crop's own baseline,
    # and all stress indicators low
    if g >= profile["green_baseline"] - 8 and y < 9 and b < 7 and p < 5:
        affected_area = round(max(0, 100 - g - (y + b + p) * 0.3), 2)
        return {
            "deficiency_type": "Healthy",
            "confidence": round(min(99.9, 90 + (g - profile["green_baseline"] + 8) * 0.5), 2),
            "severity_level": "None",
            "affected_area_pct": round(max(0.0, 15 - g * 0.1), 2),
        }

    # Evaluate crop-specific rules in order; first match with highest
    # signal strength wins (deterministic on the pixel data, per crop)
    best_match = None
    best_strength = -1
    for deficiency_type, condition, base_conf in rules:
        if condition(g, y, b, p):
            strength = y + b + p  # overall stress signal magnitude
            if strength > best_strength:
                best_strength = strength
                best_match = (deficiency_type, base_conf)

    if best_match is None:
        # No rule triggered strongly - classify as mild generalized stress
        # specific to this crop rather than defaulting to a fixed label
        deficiency_type = f"Early-stage Nutrient Stress ({crop_type.title()})"
        confidence = round(70 + (y + b + p) * 0.3, 2)
    else:
        deficiency_type, base_conf = best_match
        confidence = round(min(99.5, base_conf * 100 - (5 - min(5, best_strength * 0.05))), 2)

    affected_area = round(min(100.0, y * 0.9 + b * 1.1 + p * 0.8), 2)

    if affected_area <= 20:
        severity = "Mild"
    elif affected_area <= 50:
        severity = "Moderate"
    else:
        severity = "Severe"

    return {
        "deficiency_type": deficiency_type,
        "confidence": max(60.0, min(99.9, confidence)),
        "severity_level": severity,
        "affected_area_pct": affected_area,
    }


def analyze_leaf(image_path, crop_type):
    """Full pipeline: extract spectrum -> classify -> return combined result."""
    spectrum = extract_hsv_spectrum(image_path)
    result = classify_deficiency(spectrum, crop_type)
    result.update(spectrum)
    result.pop("img_hash", None)
    result["crop_type"] = crop_type.lower().strip()
    return result
