class PandasBackend:
    def __init__(self):
        self.backend = None
        self.type = "pandas"
        self.set()

    def set(self, backend="pandas"):
        if backend not in ["pandas", "modin"]:
            print("Wrong backend specified! Usage: pd.set('pandas') or pd.set('modin')")
            print("Using default pandas backend!")
            backend = "pandas"
        self.type = backend
        if backend == "pandas":
            import pandas

            try:
                from tqdm.auto import tqdm

                tqdm.pandas()
            except ImportError:
                print("Error in tqdm.pandas()")
                print("Pandas progress is disabled")
            self.backend = pandas
        elif backend == "modin":
            from modin import pandas

            self.backend = pandas

    def get(self):
        return self.backend

    def get_type(self):
        return self.type


pd = PandasBackend()
