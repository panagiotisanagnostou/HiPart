import os


def pytest_configure():
    # Force a non-interactive backend so CI can render plots without Tk.
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
