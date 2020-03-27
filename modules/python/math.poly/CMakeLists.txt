set(TARGET_NAME math.poly-python)
set(MODULE_NAME math_poly)
set(MODULE_DEPS math.poly numpyutils)
set(PYTHON_DEPS except-python math.linear-python types-python)
set(SWIG_INPUT_FILE "source/math_poly.i")

coda_add_swig_python_module()
