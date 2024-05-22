from setuptools import setup, Extension

setup(
    name='value',
    version='1.0',
    description='Python C extension `Value` container',
    ext_modules=[Extension('value', sources=['value.c'])],
)
