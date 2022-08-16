from setuptools import setup

required_libraries = [
    "beautifulsoup4",
    "faker",
    "html2text",
    "jupyter",
    "matplotlib",
    "numpy",
    "pandas",
    "python-dateutil",
    "requests",
    "seaborn",
    "scikit-learn",
    "scipy",
    "six",
    "tensorflow"
]

setup(
    name="deep_fm",
    version="0.2.1",
    description="factorization machine with cross entropy loss where interaction effects come from deep nonlinear relu-activated embeddings and with an additional 'metric' kernal matrix.",
    url="http://github.com/jattenberg/deep_fm",
    author="jattenberg",
    author_email="josh@attenberg.org",
    license="MIT",
    packages=["deep_fm", "tests"],
    zip_safe=False,
    install_requires=required_libraries,
)
