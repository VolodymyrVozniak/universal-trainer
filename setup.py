import setuptools

import croatoan_trainer


setuptools.setup(
    name='croatoan-trainer',
    version=croatoan_trainer.__version__,
    packages=setuptools.find_packages(),
    url='https://github.com/VolodymyrVozniak/universal-trainer',
    license='',
    author='Volodymyr Vozniak',
    author_email='vozniak.v.z@gmail.com',
    description='Universal trainer and visualizator',
    install_requires=[
        'torch>=1.13.0',
        'scikit-learn>=1.2.0',
        'pandas>=1.5.2',
        'plotly>=5.7.0'
    ],
    python_requires=">=3.8",
)
