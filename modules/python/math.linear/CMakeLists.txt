coda_add_swig_python_module(
    TARGET math.linear-python
    MODULE_NAME math_linear
    MODULE_DEPS math.linear
    PYTHON_DEPS except-python types-python
    INPUT "source/math_linear.i")
