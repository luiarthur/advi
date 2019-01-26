from setuptools import setup

setup(name='advi',
      version='0.1',
      description='My personal ADVI library',
      url='http://github.com/advi',
      author='Arthur Lui',
      author_email='luiarthur@gmail.com',
      license='MIT',
      packages=['advi'],
      test_suite='nose.collector',
      tests_require=[ 'nose' ],
      install_requires=['torch'],
      zip_safe=False)
