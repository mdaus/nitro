set(TARGET_NAME "except-python")
set(MODULE_NAME "coda_except")
set(MODULE_DEPS "except")
set(SWIG_INPUT_FILE "source/except.i")

coda_add_swig_python_module(${TARGET_NAME})
