class OOVDictionary:
    def __init__(self, d):
        self.d = d

    def __getitem__(self, x):
        if x in self.d:
            return self.d[x]
        else:
            return self.d['OOV']

    def get(self, x, default=None):
        if default is None or x in self.d:
            return self[x]
        else:
            return default

    def __len__(self):
        return len(self.d)
