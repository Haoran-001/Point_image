import argparse
import pip
from subprocess import call

def download(requirements = None):
    mirrors = 'https://mirrors.bfsu.edu.cn/pypi/web/simple'
        for package in requirements:
            call('pip install ' + package + ' -i ' + mirrors, shell=True)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, help='the filename of dependency', default='requirement.txt')
args = parser.parse_args()
f = args.filename
with open(filename) as requirements:
    download(requirements)



