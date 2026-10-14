from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mata-nonverbal",
    version="1.0.0",
    author="Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, Swarup Ranjan Behera, Sishir Kalita, Arun Balaji Buduru, Rajesh Sharma",
    description="MATA: Non-Verbal Emotion Recognition via Modality-Binding Foundation Models with Optimal Transport",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gir-ish/Non-verbal",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
