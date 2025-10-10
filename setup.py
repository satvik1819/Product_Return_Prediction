from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Reads requirements from the given file and returns them as a list."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() != ""]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements

setup(
    name='PRODUCT_RETURN_PREDICTION',
    version='0.0.1',
    author='Sathvik',
    author_email='sathvikvedantham@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
