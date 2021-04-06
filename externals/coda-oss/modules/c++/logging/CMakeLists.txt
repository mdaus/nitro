set(MODULE_NAME logging)

coda_add_module(
    ${MODULE_NAME}
    VERSION 1.1
    DEPS io-c++ mt-c++ std-c++ sys-c++ coda_oss-c++)
