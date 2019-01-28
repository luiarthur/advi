from setuptools import setup, find_packages

setup(name='advi',
      version='0.2',
      description='My personal ADVI library',
      url='http://github.com/advi',
      author='Arthur Lui',
      author_email='luiarthur@gmail.com',
      license='MIT',
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=['torch'],
      zip_safe=False)
