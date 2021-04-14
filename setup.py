import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

setup(
  name='sctui',
  version='0.0.1',
  url='https://github.com/carterjfulcher/scuti',
  author='carterjfulcher',
  author_email='fulcher.carter@gmail.com',
  packages=find_packages(),
  platforms='any',
  license='MIT',
  package_data={'': ['helpers/chi2_lookup_table.npy', 'templates/*']},
  ext_modules=[],
  description="A Kalman Filter Library",
  long_description='See https://github.com/carterjfulcher/scuti',
)
