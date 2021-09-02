class Metric:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError
