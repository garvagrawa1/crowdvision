import pytest

from crowdvision.counting import count_person_detections


def test_count_is_preserved():
    assert count_person_detections(7) == 7


def test_zero_people():
    assert count_person_detections(0) == 0


def test_negative_count_is_rejected():
    with pytest.raises(ValueError):
        count_person_detections(-1)


def test_non_integer_count_is_rejected():
    with pytest.raises(TypeError):
        count_person_detections(2.5)
