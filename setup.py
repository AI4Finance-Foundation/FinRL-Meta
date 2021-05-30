from setuptools import setup, find_packages
NAME = "neofinrl"
PACKAGES = [NAME] + ["%s.%s" % (NAME, i) for i in find_packages(NAME)]
setup(name='neofinrl',
      version='0.0.1',
      packages=PACKAGES,
      install_requires=['gym', 'numpy'])
