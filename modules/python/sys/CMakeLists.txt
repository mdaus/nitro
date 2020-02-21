set(TARGET_NAME "sys-python")
set(MODULE_NAME "coda_sys")
set(MODULE_DEPS "sys")
set(SWIG_INPUT_FILE "source/sys.i")

coda_add_swig_python_module(${TARGET_NAME})
