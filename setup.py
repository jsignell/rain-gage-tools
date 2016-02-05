from setuptools import setup, find_packages

setup(name='rain',
      version='0.1',
      description='Tools for reading rain gage data files',
      url='http://github.com/jsignell/rain-gage-tools',
      author='Julia Signell',
      author_email='jsignell@gmail.com',
      license='MIT',
      packages=['rain'],
      zip_safe=False,
      # If any package contains *.r files, include them:
      package_data={'': ['*.r', '*.R']},
      include_package_data=True)
