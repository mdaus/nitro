set(TARGET_NAME XercesC)

# set XercesC configure options
# see https://xerces.apache.org/xerces-c/build-3.html
if (WIN32)
    set(EXTRA_CMAKE_ARGS
        -Dnetwork-accessor=winsock
        -Dtranscoder=windows
        -Dmutex-manager=windows
        -Dxmlch-type=char16_t
    )
else()
    set(EXTRA_CMAKE_ARGS
        -Dnetwork-accessor=socket
        -Dtranscoder=iconv
    )
endif()
list(APPEND EXTRA_CMAKE_ARGS
    -Dmessage-loader=inmemory
)

coda_add_driver(
    ${TARGET_NAME}
    "xerces-c-3.2.2.tar.gz"
    "SHA256=dd6191f8aa256d3b4686b64b0544eea2b450d98b4254996ffdfe630e0c610413"
)

# replace Xerces' CMakeLists.txt with our patched version
ExternalProject_Add_Step(${CMAKE_PROJECT_NAME}_${TARGET_NAME} patchbuild
    COMMAND cmake -E copy ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists-patched.txt
                          ${${CMAKE_PROJECT_NAME}_xercesc_SOURCE_DIR}/CMakeLists.txt
    DEPENDEES patch
    DEPENDERS configure
)

# remove problematic symlink from Xerces' CMake install step
if (UNIX AND NOT BUILD_SHARED)
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME}_${TARGET_NAME}
        COMMAND cmake -E remove "${CMAKE_INSTALL_PREFIX}/${CODA_STD_PROJECT_LIB_DIR}/libxerces-c.so"
        DEPENDS ${CMAKE_PROJECT_NAME}_${TARGET_NAME}
    )
endif()

set(XERCESC_LIBRARY "xerces-c" CACHE STRING "name of the XercesC library")
