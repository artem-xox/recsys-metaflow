from metaflow import FlowSpec, step


class RecommendationsFlow(FlowSpec):
    """
    """

    @step
    def start(self):
        """
        """
        self.next(self.data)

    @step
    def data(self):
        """
        """
        self.next(self.train)

    @step
    def train(self):
        """
        """
        self.next(self.end)

    @step
    def end(self):
        """
        """


if __name__ == "__main__":
    RecommendationsFlow()
