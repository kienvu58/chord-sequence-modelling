class ModelI:
    def fit(self, dataset):
        raise NotImplementedError()

    def log_score(self, progression):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def generate(self, min_length):
        raise NotImplementedError()

