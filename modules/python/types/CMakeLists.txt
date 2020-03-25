set(TARGET_NAME types-python)
set(MODULE_NAME coda_types)
set(MODULE_DEPS types sys)
set(PYTHON_DEPS sys-python)
set(SWIG_INPUT_FILE "source/types.i")

coda_add_swig_python_module(${TARGET_NAME})
