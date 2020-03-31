coda_add_swig_python_module(
    TARGET sio.lite-python
    MODULE_NAME sio_lite
    MODULE_DEPS sio.lite
    PYTHON_DEPS io-python sys-python except-python
    INPUT "source/sio_lite.i")
