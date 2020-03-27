set(TARGET_NAME mem-python)
set(MODULE_NAME mem)
set(MODULE_DEPS mem)
set(SWIG_INPUT_FILE "source/mem.i")

coda_add_swig_python_module()
