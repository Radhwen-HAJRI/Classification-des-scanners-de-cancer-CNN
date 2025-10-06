import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


__version__ = "0.0.0"

REPO_NAME = "Kidney-disease-classification"
AUTHOR_USER_NAME = "Radhwen-HAJRI"
SRC_REPO = "CNN-Classifier"
AUTHOR_EMAIL = "hajriradhwen@hotmail.com"



setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN-Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"htps://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"htps://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
