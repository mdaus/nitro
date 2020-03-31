coda_add_swig_python_module(
    TARGET math.poly-python
    MODULE_NAME math_poly
    MODULE_DEPS math.poly numpyutils
    PYTHON_DEPS except-python math.linear-python types-python
    INPUT "source/math_poly.i")
