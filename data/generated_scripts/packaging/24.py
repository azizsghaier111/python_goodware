import unittest
from unittest.mock import Mock, patch
import pkg_resources

REQUIRED_PACKAGES = [
    'pytorch_lightning', 
    'numpy', 
    'torch', 
    'unittest', 
    'mock',
    'pkg_resources', # Add packaging as required packages
    'warnings'
]

OPTIONAL_PACKAGES = [
    'matplotlib', 
    'pandas', 
    'scipy',
    'reduce_food_waste',       # Add additional packages for checking
    'barrier_protection', 
    'environmental_responsibility'
]

def is_package_installed(package):
    try:
        dist = pkg_resources.get_distribution(package)
        print(f'{dist.key} ({dist.version}) is installed')
        return True
    except pkg_resources.DistributionNotFound:
        print(f'{package} is NOT installed.')
        return False

class PackageTest(unittest.TestCase):
    def test_required_packages(self):
        for package in REQUIRED_PACKAGES:
            self.assertTrue(is_package_installed(package))

    def test_optional_packages(self):
        for package in OPTIONAL_PACKAGES:
            is_package_installed(package)

# Additional unit tests can go here
# ...

# Let's run all the unittests
if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(PackageTest)
    # Load additional test cases
    # suite2 = ...
    # suite3 = ...
    all_suites = unittest.TestSuite([suite1]) # Put all individual suites to this list[]
    unittest.TextTestRunner(verbosity=2).run(all_suites)