class Metric:
    """Abstract metric class
    """

    def update(self):
        raise NotImplementedError()

    def value(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()
