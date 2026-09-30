from dataclasses import dataclass


@dataclass
class Detection:
    object: str
    confidence: float
    position: str
    proximity: str


DANGER_OBJECTS = {
    "person",
    "car",
    "motorcycle",
    "bicycle",
    "bus",
    "truck",
}


def estimate_proximity(box_area: float, frame_area: float) -> str:
    ratio = box_area / frame_area

    if ratio > 0.18:
        return "very close"
    elif ratio > 0.07:
        return "close"
    elif ratio > 0.025:
        return "medium"
    else:
        return "far"


def get_navigation_message(detections):
    if not detections:
        return "The path ahead appears clear."

    # Look for objects directly ahead
    center = [
        d for d in detections
        if d.position == "center"
    ]

    # Most urgent object first
    center.sort(
        key=lambda d: (
            d.object not in DANGER_OBJECTS,
            ["very close", "close", "medium", "far"].index(d.proximity)
        )
    )

    if center:
        d = center[0]

        if d.proximity in ("very close", "close"):
            return (
                f"Warning. A {d.object} is "
                f"{d.proximity} directly ahead."
            )

        return (
            f"There is a {d.object} "
            f"{d.proximity} ahead."
        )

    # No center obstacle → directional information
    left = [d for d in detections if d.position == "left"]
    right = [d for d in detections if d.position == "right"]

    if left and not right:
        return f"A {left[0].object} is on your left."

    if right and not left:
        return f"A {right[0].object} is on your right."

    return "Objects detected around you. Proceed carefully."