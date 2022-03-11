import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='datastats',
    version='1.0',
    author='Paul Hussey',
    author_email='pthussey77@gmail.com',
    description='A package of functions and classes for doing exploratory data analysis.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    license='MIT',
    packages=['datastats'],
    install_requires=['matplotlib', 'pandas', 'scipy', 'seaborn', 'statsmodels']
    )