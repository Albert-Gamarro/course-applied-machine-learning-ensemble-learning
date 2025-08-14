from setuptools import setup, find_packages

setup(
    name="ml_ensemble_course",
    version="0.1",
    packages=find_packages(where="src"),  # search packages in src/
    package_dir={"": "src"},  # tell setuptools "src" is root
    python_requires=">=3.7",
)
