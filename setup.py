from setuptools import find_packages, setup

setup(
    name='flex_rnn',
    version='0.1',
    description='Generate cuda kernels for linear attention mechanisms like RWKV or Mamba',
    author='Johan Sokrates Wind',
    author_email='johanswi@uio.no',
    url='https://github.com/johanwind/flex_rnn',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch',
        'ninja'
    ]
)
