from pathlib import Path
from setuptools import find_packages, setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='cargan',
    description='Chunked Autoregressive GAN for Conditional Waveform Synthesis',
    version='0.0.4',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/descriptinc/cargan',
    install_requires=[
        'librosa',
        'numpy',
        'torch',
        'torchaudio',
        'torchcrepe',
        'tqdm'],
    packages=find_packages(),
    package_data={'cargan': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'speech', 'gan', 'pytorch', 'vocoder'])
