from setuptools import find_packages, setup


setup(
    name='HEA',
    packages=find_packages(),
    version='0.1dev',
    license='MIT',
    author='Anthony Correia',
    long_description=open('README.md').read(),
    description='tools for High Energy Analysis (HEA)',
    include_package_data = True,
)
