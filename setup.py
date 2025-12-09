'''A `setup.py` file is necessary in a machine-learning project **only when you want to package, install, or distribute the project like a Python library**. It allows your project to be installed using `pip install .`, which is important when the project contains multiple modules and needs a clean, import-error-free structure. It also helps manage dependencies, ensuring that all required libraries such as NumPy, Pandas, TensorFlow, or Scikit-learn are automatically installed, making the environment reproducible. In production environments—such as Docker, cloud services, or internal pipelines—`setup.py` provides versioning, consistency, and easy deployment. If you want to publish your ML code as a reusable package or share it with a team, `setup.py` becomes essential. However, for small experiments or simple scripts, it is not required; a `requirements.txt` is usually enough.
Here is a basic example of what a `setup.py` file might look like for a machine-learning project:'''


from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """Read the requirements from a file and return them as a list."""
    requirement_list : List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and not requirement != '.e':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print(f"Warning: requirements.txt not found. No dependencies will be installed.")

    return requirement_list    


setup(
    name='networksecurity',
    version='0.1.0',
    author='Ritik Gupta',
    author_email="ritiknitc@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
    description='A machine learning project for network security',
)

    
