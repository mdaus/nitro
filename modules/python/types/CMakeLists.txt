coda_add_swig_python_module(
    TARGET types-python
    MODULE_NAME coda_types
    MODULE_DEPS types sys
    PYTHON_DEPS sys-python
    INPUT "source/types.i")
