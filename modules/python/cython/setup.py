from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext_modules=[
    Extension("nitro",
              extra_compile_args=["-g"],
              extra_link_args=["-g"],
              sources=["nitro.pyx"],
              include_dirs=["../../../install/include"],
              library_dirs=["../../../install/lib"],
              runtime_library_dirs=["../../../install/lib"],
              libraries=["nitf-c", "nrt-c"]
    )
]

setup(
    name="Nitro",
    ext_modules=cythonize(ext_modules, gdb_debug=True)
)

