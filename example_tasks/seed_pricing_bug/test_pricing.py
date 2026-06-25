from pricing import discounted_price, format_total


def test_no_discount():
    assert discounted_price(100) == 100


def test_discount_applied():
    assert discounted_price(200, discount_percent=25) == 150


def test_discount_not_cached_incorrectly():
    # Calling with a different discount for the same base price must not
    # reuse a cached result from a previous call with different kwargs.
    assert discounted_price(50, discount_percent=0) == 50
    assert discounted_price(50, discount_percent=10) == 45


def test_multiple_distinct_discounts():
    assert discounted_price(80, discount_percent=50) == 40
    assert discounted_price(80, discount_percent=25) == 60


def test_format_total_rounds_up_correctly():
    assert format_total(9.996) == "$10.00"


def test_format_total_basic():
    assert format_total(12.5) == "$12.50"


def test_format_total_truncation_edge():
    assert format_total(3.004) == "$3.00"
