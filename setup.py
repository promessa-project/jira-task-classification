import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jira-task-classification",
    version="0.0.1",
    author="Shivathanu Gopirajan Chitra",
    author_email="gcshivathanu@gmail.com",
    description="Build classification models that enable to classify task to related categories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/promessa-project/jira-task-classification",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license='unlicense',
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)