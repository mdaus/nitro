set(TARGET_NAME logging-python)
set(MODULE_NAME coda_logging)
set(MODULE_DEPS logging)
set(SWIG_INPUT_FILE "source/logging.i")

coda_add_swig_python_module(${TARGET_NAME})
