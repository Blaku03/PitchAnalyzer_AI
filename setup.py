from setuptools import find_packages, setup

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="PitchAnalyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Bartek Brzyski, Marcel Gruzewski",
    description="Get statictics of a football match",
    url="https://github.com/Blaku03/PitchAnalyzer_AI.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
