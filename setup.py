from pathlib import Path
from shutil import rmtree
from setuptools import setup, Extension

for dir in ['build', 'dist']:
    if Path(dir).exists():
        rmtree(dir)

for so in Path('.').glob('value.cpython*'):
    Path(so).unlink()

setup(
    name='value',
    version='1.0',
    description='Python C extension `Value` container',
    ext_modules=[Extension('value', sources=['value.c'])],
)
