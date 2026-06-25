def format_currency(amount):
    """Format a number as a currency string with 2 decimal places, e.g. '$12.50'."""
    # BUG: truncates to 2 decimals instead of rounding, so values that should
    # round up (e.g. 9.996 -> 10.00) come out wrong (9.99).
    truncated = int(amount * 100) / 100
    return f"${truncated:.2f}"
