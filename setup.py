from setuptools import setup, find_packages

setup(
    name="recnn",
    version="0.1",
    description="A Python toolkit for Reinforced News Recommendation.",
    long_description="A Python toolkit for Reinforced News Recommendation.",
    author="Mike Watts",
    author_email="awarebayes@gmail.com",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
    url="https://github.com/awarebayes/RecNN",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3",  # Specify which pyhton versions that you want to support
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
