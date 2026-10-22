"""Nutrition5k parsing and numerical evaluation helpers for the notebook."""

import csv
import hashlib
import math
import random
import re
from pathlib import Path

TARGETS = ("calories", "protein", "fat", "carb", "mass")


def template_index(dish_id, n_templates=8):
    """Stable across processes; deliberately changes the historical hash mapping."""
    if n_templates < 1:
        raise ValueError("At least one template is required")
    seed = int.from_bytes(hashlib.sha256(dish_id.encode("utf-8")).digest()[:4], "big")
    return random.Random(seed).randrange(n_templates)


def read_indexed_csv(path, key="dish_id"):
    """Reject missing/duplicate IDs rather than silently overwriting rows."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty table: {path}")
    result = {}
    for row in rows:
        value = row.get(key)
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {key}: {value!r} in {path}")
        result[value] = row
    return result


def parse_fact_string(text):
    """Parse the six explicit fact fields; never turn a missing value into zero."""
    if not isinstance(text, str):
        raise ValueError("Fact string must be text")
    fields = {}
    for part in text.split("|"):
        key, separator, value = part.partition(":")
        key = key.strip().lower()
        if not separator or key in fields:
            raise ValueError("Malformed or duplicate fact field")
        fields[key] = value.strip()
    expected = {"ingredients", "mass", "calories", "protein", "fat", "carbohydrates"}
    if set(fields) != expected or not fields["ingredients"]:
        raise ValueError("Fact string requires ingredients and all five numeric fields")
    result = {"ingredients": fields["ingredients"]}
    for key in expected - {"ingredients"}:
        unit = "kcal" if key == "calories" else "g"
        match = re.fullmatch(r"(\d+)\s*" + unit, fields[key])
        if not match:
            raise ValueError(f"Invalid {key}: {fields[key]!r}")
        result[key] = int(match.group(1))
    return result


def fact_numbers(text):
    parsed = parse_fact_string(text)
    return {k: parsed["carbohydrates" if k == "carb" else k] for k in TARGETS}


def legacy_extract_numbers(text):
    """Original integer regex, preserved ONLY to reproduce historical metrics.

    Its generic mass fallback can mistake another gram value for mass; decimals,
    paraphrases and conflicting mentions are outside its supported grammar.
    An extraction failure is not proof that an attribute is absent from the text.
    """
    text = text.lower()

    def find(patterns):
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return None

    return {
        "calories": find([r"(\d+)\s*(?:kcal|calories)", r"(\d+)[\s-]*calorie"]),
        "protein": find([r"(\d+)\s*g?\s*(?:of\s+)?protein"]),
        "fat": find([r"(\d+)\s*g?\s*(?:of\s+)?fat"]),
        "carb": find([r"(\d+)\s*g?\s*(?:of\s+)?(?:carbohydrates|carbs)"]),
        "mass": find([r"(\d+)\s*g(?:ram)?(?:\s+(?:dish|plate|serving|meal|food|item))",
                      r"\((\d+)g\)", r"(\d+)g\b"]),
    }


def numeric_metrics(rows, ground_truth):
    """Historical NutAcc and extracted-only MAE, with explicit denominators."""
    if not rows:
        raise ValueError("Cannot evaluate an empty sample")
    seen = set()
    ok = dict.fromkeys(TARGETS, 0)
    missing = dict.fromkeys(TARGETS, 0)
    wrong = dict.fromkeys(TARGETS, 0)
    errors = {k: [] for k in TARGETS}
    for row in rows:
        dish = row["dish_id"]
        if dish in seen or dish not in ground_truth:
            raise ValueError(f"Duplicate dish or missing ground truth: {dish}")
        seen.add(dish)
        predicted = legacy_extract_numbers(row["generated_caption"])
        supplied = fact_numbers(row["fact_string"])
        for key in TARGETS:
            truth = float(ground_truth[dish][key])
            if not math.isfinite(truth) or truth < 0:
                raise ValueError(f"Invalid ground truth: {dish}/{key}")
            value = predicted[key]
            if value is None:
                missing[key] += 1
            else:
                errors[key].append(abs(value - truth))
                if abs(value - supplied[key]) <= 1:
                    ok[key] += 1
                else:
                    wrong[key] += 1
    n = len(rows)
    metrics = {f"NutAcc_{k}": round(100 * ok[k] / n, 2) for k in TARGETS}
    metrics["NutAcc_avg"] = round(100 * sum(ok.values()) / (n * len(TARGETS)), 2)
    metrics.update({f"MAE_{k}": round(sum(errors[k]) / len(errors[k]), 2)
                    if errors[k] else None for k in TARGETS})
    metrics.update({f"Omit_{k}": round(100 * missing[k] / n, 1) for k in TARGETS})
    return {"n": n, "metrics": metrics, "unparsed_counts": missing,
            "out_of_tolerance_counts": wrong,
            "mae_denominators": {k: len(errors[k]) for k in TARGETS}}


def parse_nutrition5k_csv(filepath):
    """Parse the observed raw format: six dish fields + seven per ingredient.

    Uses csv.reader for quoted ingredient names and fails on malformed rows.
    """
    records = {}
    with Path(filepath).open(newline="", encoding="utf-8-sig") as stream:
        for line, parts in enumerate(csv.reader(stream), 1):
            if not parts:
                continue
            if len(parts) < 6 or (len(parts) - 6) % 7:
                raise ValueError(f"Unexpected Nutrition5k schema at {filepath}:{line}")
            dish = parts[0].strip()
            if not dish or dish in records:
                raise ValueError(f"Missing/duplicate dish at {filepath}:{line}")
            values = list(map(float, parts[1:6]))
            ingredients = []
            for offset in range(6, len(parts), 7):
                item = dict(zip(("grams", "calories", "fat", "carb", "protein"),
                                map(float, parts[offset + 2:offset + 7])))
                if not all(math.isfinite(v) and v >= 0 for v in item.values()):
                    raise ValueError(f"Invalid ingredient values at {filepath}:{line}")
                ingredients.append({"id": parts[offset].strip(),
                                    "name": parts[offset + 1].strip(), **item})
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"Invalid dish values at {filepath}:{line}")
            records[dish] = dict(zip(("calories", "mass", "fat", "carb", "protein"), values))
            records[dish].update(n_ingr=len(ingredients), ingredients=ingredients)
    if not records:
        raise ValueError(f"Empty Nutrition5k metadata: {filepath}")
    return records
