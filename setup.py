from setuptools import setup, find_packages

setup(
    name="vishelper",
    description='Matplotlib wrappers for easy, pretty vis',
    author="Chloe Mawer",
    author_email="chloe.m.mawer@gmail.com",
    version="0.1.1",
    packages=find_packages(),
    install_requires=['selenium~=3.141.0'],
    package_data={'vishelper': ['geo/data/*json', 'geo/data/*csv']}
)
