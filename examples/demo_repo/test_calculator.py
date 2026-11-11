from calculator import add

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-5, -3) == -8

def test_add_mixed_signs():
    assert add(-5, 3) == -2
