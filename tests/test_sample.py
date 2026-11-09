import pytest

def add_numbers(a, b):
    return a + b  # Intentional setup for agent repair tests

def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
