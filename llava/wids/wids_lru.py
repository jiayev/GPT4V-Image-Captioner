from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int, release_handler=None):
        """Initialize a new LRU cache with the given capacity."""
        self.capacity = capacity
        self.cache = OrderedDict()
        self.release_handler = release_handler

    def __getitem__(self, key):
        """Return the value associated with the given key, or None."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        """Associate the given value with the given key."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            key, value = self.cache.popitem(last=False)
            if self.release_handler is not None:
                self.release_handler(key, value)

    def __delitem__(self, key):
        """Remove the given key from the cache."""
        if key in self.cache:
            if self.release_handler is not None:
                value = self.cache[key]
                self.release_handler(key, value)
            del self.cache[key]

    def __len__(self):
        """Return the number of entries in the cache."""
        return len(self.cache)

    def __contains__(self, key):
        """Return whether the cache contains the given key."""
        return key in self.cache

    def items(self):
        """Return an iterator over the keys of the cache."""
        return self.cache.items()

    def keys(self):
        """Return an iterator over the keys of the cache."""
        return self.cache.keys()

    def values(self):
        """Return an iterator over the values of the cache."""
        return self.cache.values()

    def clear(self):
        for key in list(self.keys()):
            value = self.cache[key]
            if self.release_handler is not None:
                self.release_handler(key, value)
            del self[key]

    def __del__(self):
        self.clear()
