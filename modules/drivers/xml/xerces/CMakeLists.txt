set(TARGET_NAME "XercesC")

#set(DRIVER_FILE "xerces-c-3.1.1.tar")
#set(DRIVER_FILE_HASH "SHA256=1f57a27dd1b3806df42d5ae638c362358cd16519578d7d395fc77ffd35f4dbcb")
# Build as a completely external target
#coda_add_driver("${TARGET_NAME}" "${DRIVER_FILE}" "${DRIVER_FILE_HASH}" FALSE)

set(DRIVER_FILE "xerces-c-3.2.2.tar.gz")
set(DRIVER_FILE_HASH "SHA256=dd6191f8aa256d3b4686b64b0544eea2b450d98b4254996ffdfe630e0c610413")
coda_add_driver("${TARGET_NAME}" "${DRIVER_FILE}" "${DRIVER_FILE_HASH}" TRUE)

target_include_directories(xerces-c PUBLIC
    $<BUILD_INTERFACE:${coda-oss_xerces-c_SOURCE_DIR}/src>)
