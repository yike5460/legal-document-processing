"""
Setup script for Legal Document Processing System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="legal-document-processor",
    version="2.0.0",
    author="Legal Tech Solutions",
    author_email="support@legal-processor.com",
    description="AI-powered legal document processing system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/legal-tech/document-processor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "Topic :: Legal :: Document Processing",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "legal-processor=src.pipeline.main_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.txt"],
    },
)