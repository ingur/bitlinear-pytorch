from setuptools import find_packages, setup

setup(
    name="bitlinear-pytorch",
    packages=find_packages(exclude=["tests", "examples"]),
    version="0.0.1",
    license="MIT",
    description="Implementation of the BitLinear layer from: The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits",
    long_description_content_type="text/markdown",
    author="Ingur Veken",
    author_email="ingurv99@gmail.com",
    url="https://github.com/ingur/bitlinear-pytorch",
    keywords=[
        "pytorch",
        "quantization",
        "bitlinear",
        "linear",
        "layer",
        "ternary",
        "binary",
        "quantized",
        "quantized weights",
    ],
    install_requires=["torch>=1.10"],
    setup_requires=["pytest-runner"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
