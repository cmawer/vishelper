from setuptools import setup, find_packages

setup(
    name="vishelper",
    description='Matplotlib wrappers for easy, pretty vis',
    author="Chloe Mawer",
    author_email="chloe.m.mawer@gmail.com",
    version="0.1.0",
    packages=find_packages(),
    package_data={'vishelper': ['geo/*json', 'geo/*csv']}
)
