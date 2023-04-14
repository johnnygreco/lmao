from setuptools import find_packages, setup

__version__ = "0.0.1-beta.6"

with open("requirements.txt", "r") as f:
    install_requires = [line.replace("==", ">=") for line in f.read().splitlines() if line != "" if line[0] != "#"]

extras_require = {}

with open("requirements-dev.txt", "r") as f:
    extras_require["dev"] = [line for line in f.read().splitlines() if line != "" if line[0] != "#"]

with open("requirements-docs.txt", "r") as f:
    extras_require["docs"] = [line for line in f.read().splitlines() if line != "" if line[0] != "#"]

extras_require["all"] = [req for req_list in extras_require.values() for req in req_list]


setup(
    name="lmao-nlp",
    version=__version__,
    description="LMAO: Language Model Adapters and Orchestrators",
    author="Johnny Greco",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    url="https://github.com/johnnygreco/lmao",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
