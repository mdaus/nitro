set(MODULE_NAME sys)

coda_generate_module_config_header(${MODULE_NAME})

coda_add_module(
    ${MODULE_NAME}
    VERSION 1.2
    DEPS config-c++ except-c++ str-c++)

