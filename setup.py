"""
Setup script for BMA MIL Classifier with Domain Adaptation package
"""

from setuptools import setup, find_packages
import os

# Read README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version from config if available
version = "1.0.0"
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'configs'))
    import config
    version = getattr(config, 'VERSION', '1.0.0')
except ImportError:
    pass

setup(
    name="bma-mil-domain-adaptation",
    version=version,
    author="Research Team",
    author_email="your.email@example.com",
    description="Multi-Level Multiple Instance Learning for BMA Classification with Domain Adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/domain-adapt_trae",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/domain-adapt_trae/issues",
        "Documentation": "https://github.com/yourusername/domain-adapt_trae/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/domain-adapt_trae",
    },
    packages=find_packages(exclude=["test*", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bma-train=scripts.train:main",
            "bma-eval-kfold=scripts.evaluate_kfold:main",
            "bma-show-folds=scripts.show_fold_splits:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.py", "*.txt", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "multiple instance learning",
        "domain adaptation",
        "machine learning",
        "computer vision",
        "medical imaging",
        "histopathology",
        "deep learning",
        "attention mechanism",
    ],
)
