set(TARGET_NAME "math.linear-python")
set(MODULE_NAME "math_linear")
set(MODULE_DEPS "math.linear")
set(PYTHON_DEPS "except-python" "types-python")
set(SWIG_INPUT_FILE "source/math_linear.i")

coda_add_swig_python_module(${TARGET_NAME})
