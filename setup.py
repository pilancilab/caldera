from setuptools import find_packages, setup

VERSION = "0.0.1.dev0"
## TODO: add install_requirements
setup(
    name="lplr_llm",
    version=VERSION,
    url="https://github.com/NSagan271/winter24-lplr-extension",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={},
    python_requires=">=3.8.0"
)
