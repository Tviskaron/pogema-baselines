class HashableDict(dict):
    def __hash__(self):
        result = []
        for key, value in self.items():
            if isinstance(value, dict):
                value = HashableDict(value)
            result.append(tuple([key, value]))
        return hash(tuple(result))
