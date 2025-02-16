from setuptools import setup, find_packages

setup(
    name="vectordb",
    version="0.1.0",
    author="Christopher von Klitzing",
    author_email="",
    description="Lightweight vector database for smaller projects and experiments",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ChristophervonKlitzing/lightweight_vector_database",  # Change to your actual repository
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Add dependencies here, e.g.:
        "numpy",
    ],
)
