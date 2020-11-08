from setuptools import setup

setup(
   name='my_meep',
   version='1.0',
   description='Package that perform meep simulation',
   author='Peter Chen',
   author_email='chenduona@gmail.com',
   packages=['my_meep'],  #same as name
   install_requires=[], #external packages as dependencies
)