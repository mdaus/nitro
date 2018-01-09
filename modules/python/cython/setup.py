from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules=[
    Extension("nitro",
              extra_compile_args=["-g"],
              extra_link_args=["-g"],
              sources=["nitro.pyx"],
              include_dirs=["../../../install/include", numpy.get_include()],
              library_dirs=["../../../install/lib"],
              runtime_library_dirs=["../../../install/lib"],
              libraries=["nitf-c", "nrt-c"]
    )
]

setup(
    name="Nitro",
    version="0.1.dev-2",
    ext_modules=cythonize(ext_modules, gdb_debug=True)
)

