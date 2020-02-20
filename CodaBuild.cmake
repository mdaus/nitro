# filter_files()- Utility to filter a list of files.
#
# dest_name     - Destination variable name in parent's scope
# file_list     - Input list of files (possibly with paths)
# filter_list   - Input list of files to filter out
#                   (must be bare filenames; no paths)
#xxx Should this be a macro instead of a function?
function(filter_files dest_name file_list filter_list)
    foreach(test_src ${file_list})
        get_filename_component(test_src_name ${test_src} NAME)
        if (NOT ${test_src_name} IN_LIST filter_list)
            list(APPEND good_names ${test_src})
        endif()
    endforeach()
    set(${dest_name} ${good_names} PARENT_SCOPE)
endfunction()


# coda_add_driver() - Add a driver (3rd-party library) to the build.
#
# driver_name       - Name of the driver
# driver_url        - Location of the driver.  This can be a path relative to
#                     ${CMAKE_CURRENT_SOURCE_DIR}, or a URL.
# url_hash          - hash signature in the form hashtype=hashvalue
# import_targets    - If the 3P uses CMake, should its targets be brought into this one?
#                     If this is TRUE, the caller will need to use the targets and
#                     variables from the 3P's CMakeLists.txt to set dependencies, etc.
#                     If this is FALSE, this will create a target with the name
#                     ${CMAKE_PROJECT_NAME}_${driver_name}.
#
#  The 3P's source and build directories will be stored in
#       ${${target_name_lc}_SOURCE_DIR}, and
#       ${${target_name_lc}_BINARY_DIR} respectively,
#
#       where ${target_name_lc} is the lower-cased target name.
include(FetchContent) # Requires CMake 3.11+
include(ExternalProject)
function(coda_add_driver driver_name driver_file driver_hash import_targets)
    set(target_name ${CMAKE_PROJECT_NAME}_${driver_name})
    # Use 'FetchContent' to download and unpack the files.  Set it up here.
    #xxx This URL could be changed to fetch from the official download site if desired.
    #xxx This function could be improved to allow an absolute path.
    FetchContent_Declare(${target_name}
        URL "${CMAKE_CURRENT_SOURCE_DIR}/${driver_file}"
        URL_HASH ${driver_hash}
    )
    FetchContent_GetProperties(${target_name})
    # The returned properties use the lower-cased name
    string(TOLOWER ${target_name} target_name_lc)
    if (NOT ${target_name_lc}_POPULATED) # This makes sure we only fetch once.
        message("Populating content for external dependency ${driver_name}")
        # Now (at configure time) unpack the content.
        FetchContent_Populate(${target_name})
        # Remember where we put stuff
        set("${target_name_lc}_SOURCE_DIR" "${${target_name_lc}_SOURCE_DIR}" CACHE INTERNAL "source directory for ${target_name_lc}")
        set("${target_name_lc}_BINARY_DIR" "${${target_name_lc}_BINARY_DIR}" CACHE INTERNAL "source directory for ${target_name_lc}")
        # Queue a build for build-time.
        #xxx Should we change the build directory to match that of other projects?
        if (EXISTS "${${target_name_lc}_SOURCE_DIR}/CMakeLists.txt")
            # Found a CMakeLists.txt.  Configure with CMake.
            if (import_targets)
                # Bring the external project's targets into our own configuration process.
                add_subdirectory(${${target_name_lc}_SOURCE_DIR} ${${target_name_lc}_BINARY_DIR})
            else()
                # Build as an external target
                ExternalProject_Add(${target_name}
                    SOURCE_DIR "${${target_name_lc}_SOURCE_DIR}"
                    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
                    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                )
            endif()
        elseif (EXISTS "${${target_name_lc}_SOURCE_DIR}/configure")
            # No CMakeLists.txt, but found a configure script.
            ExternalProject_Add(${target_name}
                SOURCE_DIR "${${target_name_lc}_SOURCE_DIR}"
                INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
                CONFIGURE_COMMAND cmake -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} sh <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
                BUILD_COMMAND $(MAKE)
                INSTALL_COMMAND $(MAKE) install
            )
        else()
            message("Driver ${driver_name} unpacked to ${${target_name_lc}_SOURCE_DIR}, but no configuration method foun
d.")
        endif()
    endif()
endfunction()


# coda_add_tests()  - Add a module's tests or unit tests to the build
#
# module_name       - Name of the module
# dir_name          - Subdirectory containing the tests' source code
#                     All source files beneath this directory will be used.
#                     Each source file is assumed to create a separate executable.
# deps              - Modules that the tests are dependent upon.
# filter_list       - Source files to ignore
function(coda_add_tests module_name dir_name deps extra_deps filter_list)
    # Find all the source files, relative to the module's directory
    file(GLOB_RECURSE local_tests RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${dir_name}/*.cpp")
    # Filter out ignored files
    filter_files(local_tests "${local_tests}" "${filter_list}")

    set(test_group_tgt "${module_name}_tests")
    if (NOT TARGET ${test_group_tgt})
        add_custom_target(${test_group_tgt})
    endif()

    if (MSVC)
        add_compile_options(/W3) # change this to /W4 later
    endif()

    foreach(test_src ${local_tests})
        # Use the base name of the source file as the name of the test
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} "${test_src}")
        add_dependencies(${test_group_tgt} ${test_name})
        get_filename_component(test_dir ${test_src} DIRECTORY)
        # Do a bit of path manipulation to make sure tests in deeper subdirs retain those subdirs in their build outputs
#xxxTODO double-check this
        file(RELATIVE_PATH test_subdir "${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}" "${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}")
        # message(STATUS "Generating Test: module_name=${module_name} test_src=${test_src}  test_name=${test_name}  test_dir=${test_dir}  test_subdir= ${test_subdir} deps=${deps}")
        # Set IDE subfolder so that tests appear in their own tree
        set_target_properties("${test_name}" PROPERTIES FOLDER "${dir_name}/${module_name}/${test_subdir}")

        # Add our output directory to the include path, to pick up 3p headers.
        target_include_directories("${test_name}" PUBLIC "${CMAKE_PREFIX_PATH}/${CODA_STD_PROJECT_INCLUDE_DIR}")

#xxx This shouldn't be needed; it should come from the module.
        # We need the parent directory include for TestCase.h
        target_include_directories("${test_name}"  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../${CODA_STD_PROJECT_INCLUDE_DIR}")
#xxx This shouldn't be needed; it should come from the module.
#       target_include_directories(${test_name}  PUBLIC "${CODA_STD_PROJECT_INCLUDE_DIR}")
        # Automatically depend on the parent module, plus any others that were specified
        #message(STATUS "target_link_libraries(${test_name} ${module_name} ${deps})")
        target_link_libraries("${test_name}" ${module_name} ${deps})

#xxx This should also not be needed; if our target depends on it, we should too.
        if (extra_deps)
            add_dependencies("${test_name}" ${extra_deps})
        endif()

        # Install [unit]tests to separate subtrees
        install(TARGETS ${test_name} RUNTIME DESTINATION "${dir_name}/${module_name}/${test_subdir}")
    endforeach()
endfunction()


# coda_add_library_impl() - Add a library to the build
#
# tgt_name          - Name of the module
# tgt_lang          - Language of the library
# tgt_deps          - List of linkable dependencies for the library
# tgt_extra_deps    - List of non-linkable dependencies for the library
# source_filter     - Source files to ignore
function(coda_add_library_impl tgt_name tgt_lang tgt_deps tgt_extra_deps source_filter)
    set(local_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR})

    # Find all the source files, relative to the module's directory
    file(GLOB_RECURSE local_sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CODA_STD_PROJECT_SOURCE_DIR}/*.cpp")

    # Filter out ignored files
    filter_files(local_sources "${local_sources}" "${source_filter}")

    # Libraries without sources must be declared to CMake as INTERFACE libraries
    if (NOT local_sources)
        set(lib_type "INTERFACE")  # No sources; make it an INTERFACE library
        set(header_type "INTERFACE")
    else()
        set(lib_type "")  # Allow default
        set(header_type "PUBLIC")
    endif()

    if (MSVC)
        add_compile_options(/W3) # change this to /W4 later
    endif()

    add_library("${tgt_name}" ${lib_type} ${local_sources})

    # Periods in target names for dirs are replaced with slashes (subdirectories).
    string(REPLACE "." "/" tgt_munged_dirname ${tgt_name})

    # Periods in target names for files are replaced with underscores.
    # Note that this variable name is used in the *.cmake.in files.
    string(REPLACE "." "_" tgt_munged_name ${tgt_name})

    # Find all the header files, relative to the module's directory, and add them.
    file(GLOB_RECURSE local_headers RELATIVE ${local_include_dir} *.h)
    #message("tgt_name=${tgt_name}  local_headers=${local_headers}")

    #set(build_interface_headers ${local_headers})
    #list(TRANSFORM build_interface_headers PREPEND "${local_include_dir}/")
    #set(install_interface_headers ${local_headers})
    #list(TRANSFORM install_interface_headers PREPEND "${CODA_STD_PROJECT_INCLUDE_DIR}/")

    # If we find a *_config.h.cmake.in file, generate the corresponding *_config.h, and put the
    #   target directory in the include path.
    #xxx This should probably look for all *.cmake.in files and process them.
    set(config_file_template "${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_munged_dirname}/${tgt_name}_config.h.cmake.in")
    if (EXISTS ${config_file_template})
        # message(STATUS "Processing config header: ${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_munged_dirname}/${tgt_name}_config.h.cmake.in)")
        set(config_file_out "${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_munged_dirname}/${tgt_munged_name}_config.h")
        configure_file(${config_file_template} ${config_file_out})
        target_include_directories(${tgt_name} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}")
        #list(APPEND build_interface_headers "${CMAKE_CURRENT_BINARY_DIR}/${config_file_out}")
        #list(APPEND install_interface_headers "${config_file_out}")
    endif()

    # Associate headers with target
    #target_sources("${tgt_name}" "${header_type}"
    #    $<BUILD_INTERFACE:${build_interface_headers}>
    #    $<INSTALL_INTERFACE:${install_interface_headers}>)

    if (NOT lib_type STREQUAL "INTERFACE")
        if (tgt_lang)
            set_target_properties("${tgt_name}" PROPERTIES OUTPUT_NAME "${tgt_name}-${tgt_lang}")
        endif()
        if (tgt_deps)
            target_link_libraries("${tgt_name}" ${tgt_deps})
        endif()
        if (tgt_extra_deps)
            add_dependencies("${tgt_name}" ${tgt_extra_deps})
        endif()
    else()
        if (tgt_deps)
            target_link_libraries("${tgt_name}" INTERFACE ${tgt_deps})
        endif()
    endif()

    # Add include directories
    target_include_directories(${tgt_name} ${header_type}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CODA_STD_PROJECT_INCLUDE_DIR}>)

    # Add our output directory to the include path, to pick up 3p headers.
    #target_include_directories("${tgt_name}" ${header_type} "${CMAKE_PREFIX_PATH}/${CODA_STD_PROJECT_INCLUDE_DIR}")


    # Set up install destinations for binaries
    install(TARGETS "${tgt_name}"
            #EXPORT "${tgt_name}_TARGETS"
            LIBRARY DESTINATION ${CODA_STD_PROJECT_LIB_DIR}
            ARCHIVE DESTINATION ${CODA_STD_PROJECT_LIB_DIR}
            RUNTIME DESTINATION ${CODA_STD_PROJECT_BIN_DIR}
    )
    # Set up install destination for headers
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}"
            DESTINATION ${CODA_STD_PROJECT_INCLUDE_DIR})

    #xxx? Perhaps change the above to:
    #install(INCLUDES DESTINATION ${CODA_STD_PROJECT_INCLUDE_DIR})

    # cannot use exports until all external dependencies have their own exports defined
    #install(EXPORT "${tgt_name}_TARGETS"
    #    FILE ${tgt_name}_TARGETS.cmake
    #    NAMESPACE ${tgt_name}::
    #    DESTINATION ${CODA_STD_PROJECT_LIB_DIR}/cmake/${tgt_name}
    #)

#[[  #xxx TODO Export the library interface? See https://www.youtube.com/watch?v=bsXLMQ6WgIk
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        VERSION ${${tgt_name}_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES "${tgt_munged_name}_config.cmake" "${tgt_munged_name}_config_version.cmake"
        DESTINATION ${CODA_STD_PROJECT_LIB_DIR/cmake/${tgt_munged_dirname}
    )
    # Then, pull in:
    include(CMakeFindDependencyMacro)
    find_dependency(mydepend 1.0) # Version#
    include("${CMAKE_CURRENT_LIST_DIR}/${tgt_munged_name}_TARGETS.cmake")

    #xxx Note that it will probably also necessary to add the $<INSTALL_INTERFACE:include> to the target include dirs.

    #xxx Also, add_library("${tgt_name}::${tgt_name}" ALIAS ${tgt_name})
#]]
endfunction()


# Add a library and its associated tests to the build.
#
#   This is a wrapper function to facilitate calls to the above routines from sub-projects.
#
#   To simplify things for callers that don't want to use many of the potential arguments,
#     this method does not take formal parameters.  Instead, callers should set any of the
#     following variables to define the library:
#
#       TARGET_LANG     Language of the library
#       MODULE_DEPS     List of dependencies for the library
#       EXTRA_DEPS      List of non-linkable dependencies for the library
#       SOURCE_FILTER   Source files to ignore
#
#   Directories defined by the variables CODA_STD_PROJECT_TESTS_DIR and
#     CODA_STD_PROJECT_UNITTESTS_DIR will be searched for source files; each of these
#     will be compiled into test executables. The following variables affect the test
#     executable creation:
#
#       TEST_DEPS       - List of dependencies for the files under CODA_STD_PROJECT_TESTS_DIR
#       TEST_FILTER     - List of source files to ignore under CODA_STD_PROJECT_TESTS_DIR
#       UNITTEST_DEPS   - List of dependencies for the files under CODA_STD_PROJECT_UNITTESTS_DIR
#       UNITTEST_FILTER - List of source files to ignore under CODA_STD_PROJECT_UNITTESTS_DIR
#
#  The caller can then simply call coda_add_library(target_name)
#
function(coda_add_library tgt_name)
    coda_add_library_impl("${tgt_name}" "${TARGET_LANG}"
                          "${MODULE_DEPS}" "${EXTRA_DEPS}" "${SOURCE_FILTER}")
    if (CODA_BUILD_TESTS)
        coda_add_tests("${tgt_name}" "${CODA_STD_PROJECT_TESTS_DIR}"
                       "${TEST_DEPS}" "${EXTRA_DEPS}" "${TEST_FILTER}")
        coda_add_tests("${tgt_name}" "${CODA_STD_PROJECT_UNITTESTS_DIR}"
                       "${UNITTEST_DEPS}" "${EXTRA_DEPS}" "${UNITTEST_FILTER}")
    endif()
endfunction()
