set(TARGET_NAME io-python)
set(MODULE_NAME coda_io)
set(MODULE_DEPS io)
set(SWIG_INPUT_FILE "source/io.i")

coda_add_swig_python_module()
