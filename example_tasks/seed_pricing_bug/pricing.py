from cache import memoize
from formatting import format_currency


@memoize
def discounted_price(base_price, discount_percent=0):
    """Return the price after applying discount_percent off base_price."""
    return base_price * (1 - discount_percent / 100)


def format_total(amount):
    return format_currency(amount)
