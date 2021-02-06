import argparse
from subprocess import call
import download_pip_dependency

def download(packages = None):
    for package in packages:
        call("conda install " + package, shell=True)


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default='point_image.yaml')
args = parser.parse_args()
filename = args.filename
with open filename as f:
    contents = yaml.load(f)
    conda_dependencies = [item for item in x['dependencies'] if not isinstance(item, dict)]
    pip_dependencies = [item for item in contents['dependencies'] if isinstance(item, dict)][0]['pip']
download(conda_dependencies)
download_pip_dependency.download(pip_dependencies)