class TemporalKVCache:
    def __init__(self, max_length):
        """
        Initializes a queue for temporal Key-Value cache.
        Args:
            max_length: Maximum number of conditional frames allowed in the cache.
        """
        self.max_length = max_length
        self.cache = []

    def update_cache(self, new_kv_pairs):
        """
        Updates the temporal cache with freshly computed KV pairs.
        Args:
            new_kv_pairs (list): Newly computed key-value pairs to add to the cache.
        """
        self.cache.extend(new_kv_pairs)
        # Remove earliest KV pairs if cache exceeds max_length
        while len(self.cache) > self.max_length:
            self.cache.pop(0)

    def read_cache(self):
        """
        Reads the current KV pairs in the cache.
        Returns:
            List of KV pairs.
        """
        return self.cache

class SpatialKVCache:
    def __init__(self):
        """
        Initializes a reduced spatial Key-Value cache for prefix enhancement.
        """
        self.cache = None

    def overwrite_cache(self, current_chunk):
        """
        Overwrites spatial cache with key-value from the current chunk.
        Args:
            current_chunk: Key-value pair derived from the most recently generated frames.
        """
        self.cache = current_chunk

    def read_cache(self):
        """
        Reads the current spatial KV cache.
        Returns:
            Current spatial key-value pairs.
        """
        return self.cache
