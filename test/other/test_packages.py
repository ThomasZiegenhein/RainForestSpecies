""" test packages and dependencies for this project """

SUCCESS_STRING = "======="
FAIL_STRING = "-------"

print("Entering package testing, success string: " + SUCCESS_STRING + " fail string: " + FAIL_STRING)

try:
    import sys
    import logging
    import importlib
    import subprocess
    import pkg_resources
    import random
    import os
except:
    print("cannot import essentials")
    exit()

packages = ["numpy", "mxnet", "librosa", "pandas", "matplotlib", "seaborn", "interval", "gluoncv"]

for pkg in packages:
    try:
        __import__(pkg, globals=globals())
        print(SUCCESS_STRING + "imported: " + pkg + ": " + pkg_resources.get_distribution(pkg).version)
    except ImportError as e:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except:
            logging.error(FAIL_STRING + "pip install failed: " + pkg)
