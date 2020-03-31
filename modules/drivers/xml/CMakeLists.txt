set(XML_HOME "" CACHE STRING "path to pre-existing XML library installation, \
                              if not provided the XML library will be built")

set(CODA_XML_LAYER "XercesC" CACHE STRING "XML backend to use")
set("USE_${CODA_XML_LAYER}" TRUE CACHE INTERNAL "")

add_subdirectory("xerces")
