set(TARGET_NAME "sio.lite-python")
set(MODULE_NAME "sio_lite")
set(MODULE_DEPS "sio.lite")
set(PYTHON_DEPS "io-python" "sys-python" "except-python")
set(SWIG_INPUT_FILE "source/sio_lite.i")

coda_add_swig_python_module(${TARGET_NAME})
