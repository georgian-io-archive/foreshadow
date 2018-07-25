from setuptools import setup, find_packages

long_description = """
TODO
"""

setup(
    name="foreshadow",
    version="0.0.1",
    url="https://github.com/georgianpartners/foreshadow",
    author="Adithya Balaji, Alexander Allen",
    author_email="",
    license="Apache-2.0",
    description="Peer into the future of a data science project",
    long_description=long_description,
    packages=find_packages(),
    zip_safe=True,
    install_requires=[
        "numpy>=1.14.5",
        "pandas>=0.23.3",
        "Cython>=0.28.4",
        "scikit-learn==0.19.2",
        "scipy>=1.1.0",
        "sklearn==0.0",
        "auto_sklearn>=0.4.0",
        "TPOT>=0.9.3",
    ],
    extras_require={
        "tests": [
            "black>=18.6b4",
            "coverage>=4.5.1",
            "coveralls>=1.3.0",
            "pip-autoremove>=0.9.1",
            "pytest>=3.6.2",
            "pytest-cov>=2.5.1",
            "tox>=3.0.0",
            "tox-travis>=0.10",
        ]
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["data science", "machine learning", "feature engineering", "automl"],
)
