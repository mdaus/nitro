from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

opts = {'extra_compile_args': ["-g"],
        'extra_link_args': ["-g"],
        'include_dirs': ["../../../install/include", numpy.get_include()],
        'library_dirs': ["../../../install/lib"],
        'runtime_library_dirs': ["../../../install/lib"],
        'libraries': ["nitf-c", "nrt-c"],
       }

ext_modules = [
    Extension("nitro.nitro", sources=["nitro.pyx"], **opts),
    Extension("nitro.dataextension_segment", sources=["dataextension_segment.pyx"], **opts),
    Extension("nitro.error", sources=["error.pyx"], **opts),
    Extension("nitro.field", sources=["field.pyx"], **opts),
    Extension("nitro.header", sources=["header.pyx"], **opts),
    Extension("nitro.tre", sources=["tre.pyx"], **opts),
    Extension("nitro.types", sources=["types.pyx"], **opts),
]

setup(
    name="nitro",
    version="0.2dev1",
    packages=["nitro"],
    ext_modules=cythonize(ext_modules, gdb_debug=True)
)

