try:
    from setuptools import setup #enables develop
except ImportError:
    from distutils.core import setup

install_requires = [
    'torch',
    'pandas',
    'scipy',
    'spacy',
    'tqdm',
    'transformers',
    'nltk',
    'tqdm',
    'torchvision',
    'setuptools',
    'numpy',
    'scipy',
    'scikit_learn'
]

setup(
    name='webert',
    version='0.0.1',
    description='Compute dynamic and static BERT embeddings',
    author='P.A: Perez-Toro',
    author_email='paula.perezt@udea.edu.co',
    url='https://github.com/PauPerezT/WEBERT/',
    download_url='https://github.com/PauPerezT/WEBERT//archive/0.0.1.tar.gz',
    license='apache',
    install_requires=install_requires,
    packages=['webert'],
    package_data={'': ['texts/*']},
    keywords = ['word embeddings', 'bert', 'transoformers'],
    dependency_links=['git+git://github.com/huggingface/transformers'],
    classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: Apache License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],

)


