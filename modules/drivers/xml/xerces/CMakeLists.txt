set(TARGET_NAME "XercesC")

if (WIN32)
    set(EXTRA_CMAKE_ARGS
        -Dnetwork-accessor=winsock
        -Dtranscoder=windows
        -Dmutex-manager=windows
        -Dxmlch-type=char16_t
    )
endif()

coda_add_driver(
    "${TARGET_NAME}"
    "xerces-c-3.2.2.tar.gz"
    "SHA256=dd6191f8aa256d3b4686b64b0544eea2b450d98b4254996ffdfe630e0c610413"
)

if (WIN32)
    set(XERCESC_LIBRARY "xerces-c_3" CACHE STRING "name of the XercesC library")
else()
    set(XERCESC_LIBRARY "xerces-c" CACHE STRING "name of the XercesC library")
endif()
