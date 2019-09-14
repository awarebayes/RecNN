from setuptools import setup, find_packages

setup(name='recnn',
      version='0.1',
      description="A Python toolkit for Reinforced News Recommendation.",
      long_description="A Python toolkit for Reinforced News Recommendation.",
      author='Mike Watts',
      author_email='awarebayes@gmail.com',
      license='Apache 2.0',
      packages=find_packages(),
      install_requires=[
          'torch',
          'numpy',
          'torchvision'
      ],
      zip_safe=False
      )
