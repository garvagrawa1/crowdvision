"""Small, testable helpers for CrowdVision."""


def count_person_detections(number_of_detections: int) -> int:
    """Return a safe non-negative count of detected person boxes."""
    if not isinstance(number_of_detections, int):
        raise TypeError("number_of_detections must be an integer")
    if number_of_detections < 0:
        raise ValueError("number_of_detections cannot be negative")
    return number_of_detections
