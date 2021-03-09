import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="geotransfer",
    version="0.0.0",
    author="Kris & Xinran",
    author_email="",
    description="Package for geographic transfer learning experiments on DeepGlobe",
    url="https://github.com/XinranMiao/Transfer_Learning_Remote_Sensing",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
