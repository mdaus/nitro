set(TARGET_NAME mt-python)
set(MODULE_NAME mt)
set(MODULE_DEPS mt sys)
set(PYTHON_DEPS except-python mem-python types-python)
set(SWIG_INPUT_FILE "source/mt.i")

coda_add_swig_python_module()
