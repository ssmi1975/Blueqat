import setuptools

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("blueqat/_version.py", "r", encoding="utf-8") as f:
    exec(f.read())

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requirements = list(map(str.strip, f))

setuptools.setup(
    name = "blueqat",
    version=__version__,
    author="MDR Inc.",
    author_email="kato@mdrft.com",
    description="Quantum gate simulator",
    long_description=long_description,
    url="https://github.com/mdrft/blueqat",
    license="Apache 2",
    packages=setuptools.find_packages(),
    install_requirements=install_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
    ]
)
