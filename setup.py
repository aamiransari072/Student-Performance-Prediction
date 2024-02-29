from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This Function will return the list of requirements.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements


setup(
    name='Student-performance-predictions',
    version='0.0.1',
    author='Aamir Ansari',
    author_email='amirbxr2003@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
