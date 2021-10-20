import setuptools

setuptools.setup(
    setup_requires=['pbr'],
    pbr=False,
    packages=setuptools.find_packages(exclude=['test']),
    python_requires='>=3.7',
    install_requires=[],
)
