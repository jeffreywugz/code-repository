from distutils.core import setup, Extension

module = Extension('layout', sources=['layout.c','layout_wrapper.c'])

setup(name='layout', version='1.0',
        description='This is a graph layout package',
        ext_modules=[module])
