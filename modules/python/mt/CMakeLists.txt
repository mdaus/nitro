coda_add_swig_python_module(
    TARGET mt-python
    MODULE_NAME mt
    MODULE_DEPS mt sys
    PYTHON_DEPS except-python mem-python types-python
    INPUT "source/mt.i")
