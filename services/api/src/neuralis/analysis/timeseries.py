import math


def minmax_decimate(points: list[dict], budget: int, key: str = "loss") -> list[dict]:
    """Preserve endpoints and extrema in temporal order within each bucket."""
    if budget < 4:
        raise ValueError("Curve budget must be at least four")
    if len(points) <= budget:
        return points
    buckets = max(1, (budget - 2) // 2)
    width = (len(points) - 2) / buckets
    selected = [0]
    for bucket in range(buckets):
        start = 1 + int(bucket * width)
        end = min(len(points) - 1, 1 + int((bucket + 1) * width))
        candidates = [index for index in range(start, end) if isinstance(points[index].get(key), (int, float))
                      and math.isfinite(points[index][key])]
        if not candidates:
            continue
        low = min(candidates, key=lambda index: points[index][key])
        high = max(candidates, key=lambda index: points[index][key])
        selected.extend(sorted({low, high}))
    selected.append(len(points) - 1)
    return [points[index] for index in sorted(set(selected))]


def align_curves(first: list[dict], second: list[dict], key: str = "loss"):
    """Compare only measured common steps; no hidden interpolation."""
    lookup = {item["step"]: item for item in second}
    result = []
    for item in first:
        other = lookup.get(item["step"])
        if other is None or key not in item or key not in other:
            continue
        result.append({"step": item["step"], "first": item[key], "second": other[key],
                       "difference": other[key] - item[key], "interpolated": False})
    return result
