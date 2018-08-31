set("TARGET_NAME"		"xml.lite")
set("MAINTAINER"		"jmrandol@users.sourceforge.net")
set("TARGET_VERSION"	"1.2")
set("MODULE_DEPS"		"io" "mem" "mt" "logging")
set("UNITTEST_DEPS"		"sys")
set("TEST_FILTER"		"MMParserTest1.cpp" "MinidomParserTest2.cpp")
set("TEST_DEPS"			"cli")

set("CODA_XML_LAYER"	"XercesC")  # "EXPAT", "LibXml2", or "XercesC"

set("USE_${CODA_XML_LAYER}"	TRUE)

# A few special case bits for specific libs
if (USE_EXPAT)
	set("EXTRA_DEPS" ${EXTRA_DEPS} "${CMAKE_PROJECT_NAME}_${CODA_XML_LAYER}") #xxx Needed?
endif()
if (USE_XercesC)
	set("EXTRA_DEPS" ${EXTRA_DEPS} "${CMAKE_PROJECT_NAME}_${CODA_XML_LAYER}") #xxx Needed?
endif()
if (USE_LibXml2)
	list(APPEND CMAKE_PREFIX_PATH "/opt/anaconda/4.3.0") #xxx Hard-coded path hack
endif()

# Find the desired XML package and add its imported target to our dependencies
include("Find${CODA_XML_LAYER}")
find_package(${CODA_XML_LAYER})
if (${CODA_XML_LAYER}_FOUND)
	list(APPEND "MODULE_DEPS" "${CODA_XML_LAYER}::${CODA_XML_LAYER}")
endif()

# The imported target wasn't added to FindLibXml2 until CMake 3.12
if (USE_LibXml2)
	if("${CMAKE_VERSION}" VERSION_LESS "3.12.0")
		if(LibXml2_FOUND AND NOT TARGET LibXml2::LibXml2)
   			add_library(LibXml2::LibXml2 UNKNOWN IMPORTED)
   			set_target_properties(LibXml2::LibXml2 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_INCLUDE_DIRS}")
   			set_property(TARGET LibXml2::LibXml2 APPEND PROPERTY IMPORTED_LOCATION "${LIBXML2_LIBRARY}")
		endif()
	endif()
endif()

if (${CODA_XML_LAYER}_FOUND) # Don't bother building if we've got no implementation.
	coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

	if (UNIX AND USE_EXPAT)
		#xxx XMLReaderExpat.h is noncompliant, but ignore that for now.
		target_compile_options("${TARGET_NAME}" PUBLIC -fpermissive)
	endif()
endif()
