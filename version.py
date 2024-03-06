import importlib

# List of library names to check versions for
libraries = [
    "jax",
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "transformers",
    "optax",
    "equinox",
    "datasets",
    # Add any other libraries you're interested in
]

for lib in libraries:
    try:
        # Dynamically import the module
        module = importlib.import_module(lib)
        
        # Attempt to print the version
        print(f"{lib}: {module.__version__}")
    except ImportError as err:
        print(f"{lib}: {err}")
        # print(f"{lib}: Not installed")
    except AttributeError:
        print(f"{lib}: Version not accessible via __version__")
