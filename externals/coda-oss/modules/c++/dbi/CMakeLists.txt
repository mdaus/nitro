set(MODULE_NAME dbi)

coda_add_module(
    ${MODULE_NAME}
    VERSION 1.0
    DEPS io-c++ ${SQL_LIB}) # see modules/drivers/sql
