from glob import glob
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os

opts = {'extra_compile_args': ["-g"],
        'extra_link_args': ["-g"],
        'include_dirs': ["../../../install/include", numpy.get_include()],
        'library_dirs': ["../../../install/lib"],
        'runtime_library_dirs': ["$ORIGIN/../../.."],
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

nitro_root = os.environ.get('NITRO_ROOT', os.path.abspath('../../../install'))
if not os.path.exists("nitro/plugins"):
    print("Linking plugins into package nitro")
    os.symlink(os.path.join(nitro_root, "share/nitf/plugins"), "nitro/plugins")

setup(
    name="nitro",
    version="1.0.3",
    packages=find_packages(),
    install_requires=["numpy>=1.13.0", "Deprecated~=1.1.0"],
    include_package_data=True,
    ext_modules=cythonize(ext_modules, gdb_debug=True)
)

