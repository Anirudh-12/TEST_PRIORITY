"""
conftest.py — pytest configuration for the test suite.
Prevents pytest from collecting function-named tests in source modules.
"""
collect_ignore_glob = [
    "evaluation/*.py",
    "bugsinpy/*.py",
    "analysis/*.py",
    "environment/*.py",
    "policies/*.py",
    "training/*.py",
    "experiments/*.py",
]
