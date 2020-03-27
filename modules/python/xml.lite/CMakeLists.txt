set(TARGET_NAME xml.lite-python)
set(MODULE_NAME xml_lite)
set(MODULE_DEPS xml.lite)
set(SWIG_INPUT_FILE "source/xml_lite.i")

coda_add_swig_python_module()
