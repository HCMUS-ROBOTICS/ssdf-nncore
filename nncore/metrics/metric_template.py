class Metric:
    """Abstract metric class
    """

    def __init__(self, *args, **kwargs):
        NotImplemented

    def update(self):
        NotImplemented

    def value(self):
        NotImplemented

    def reset(self):
        NotImplemented

    def summary(self):
        NotImplemented
