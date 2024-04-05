from setuptools import find_packages, setup

# generate install_requires from requirements.txt file
install_requires = open("requirements.txt", "r").read().strip().split("\n")
print(f"install_requires:{install_requires}")


config = {
    "name": "scprinter",
    "license": "MIT",
    "include_package_data": True,
    "description": "Multiscale footprints",
    "download_url": "https://github.com/broadinstitute/PRINT",
    "version": "0.0.1a",
    "packages": find_packages(),
    "python_requires": ">=3.8",
    "install_requires": install_requires,
    "zip_safe": False,
}

if __name__ == "__main__":
    setup(**config)
