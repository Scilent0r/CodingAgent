def memoize(func):
    """Cache results of func by its arguments to avoid recomputation."""
    cache = {}

    def wrapper(*args, **kwargs):
        key = args  # BUG: ignores kwargs entirely, so calls that differ only
        # by keyword argument incorrectly reuse a stale cached result.
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper
