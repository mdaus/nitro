# filter_files()- Utility to filter a list of files.
#
# dest_name     - Destination variable name in parent's scope
# file_list     - Input list of files (possibly with paths)
# filter_list   - Input list of files to filter out
#                   (must be bare filenames; no paths)
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
# driver_hash       - hash signature in the form hashtype=hashvalue
#
#  The 3P's source and build directories will be stored in
#       ${${target_name_lc}_SOURCE_DIR}, and
#       ${${target_name_lc}_BINARY_DIR} respectively,
#
#       where ${target_name_lc} is the lower-cased target name.
include(FetchContent) # Requires CMake 3.11+
include(ExternalProject)
function(coda_add_driver driver_name driver_file driver_hash)
    set(target_name ${CMAKE_PROJECT_NAME}_${driver_name})
    # Use 'FetchContent' to download and unpack the files.  Set it up here.
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
        set("${target_name_lc}_SOURCE_DIR" "${${target_name_lc}_SOURCE_DIR}"
            CACHE INTERNAL "source directory for ${target_name_lc}")
        set("${target_name_lc}_BINARY_DIR" "${${target_name_lc}_BINARY_DIR}"
            CACHE INTERNAL "binary directory for ${target_name_lc}")
        # Queue a build for build-time.
        if (EXISTS "${${target_name_lc}_SOURCE_DIR}/CMakeLists.txt")
            # Found CMakeLists.txt
            set(target_cmake_args
                "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
                "-DBUILD_SHARED_LIBS:BOOL=OFF"
                ${EXTRA_CMAKE_ARGS})
            if (MSVC)
                # For MSVC, ExternalProject_Add needs custom install command to
                # properly set CMAKE_BUILD_TYPE
                ExternalProject_Add(${target_name}
                    SOURCE_DIR "${${target_name_lc}_SOURCE_DIR}"
                    BINARY_DIR "${${target_name_lc}_BINARY_DIR}"
                    PREFIX "${CMAKE_INSTALL_PREFIX}"
                    CMAKE_ARGS ${target_cmake_args}
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ${CMAKE_COMMAND} --build .
                        --config ${CMAKE_BUILD_TYPE}
                        --target install
                )
            else()
                list(APPEND target_cmake_args
                    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
                )
                ExternalProject_Add(${target_name}
                    SOURCE_DIR "${${target_name_lc}_SOURCE_DIR}"
                    BINARY_DIR "${${target_name_lc}_BINARY_DIR}"
                    PREFIX "${CMAKE_INSTALL_PREFIX}"
                    CMAKE_ARGS ${target_cmake_args}
                )
            endif()
        elseif (EXISTS "${${target_name_lc}_SOURCE_DIR}/configure")
            # No CMakeLists.txt, but found a configure script.
            ExternalProject_Add(${target_name}
                SOURCE_DIR "${${target_name_lc}_SOURCE_DIR}"
                INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
                CONFIGURE_COMMAND cmake -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
                BUILD_COMMAND $(MAKE)
                INSTALL_COMMAND $(MAKE) install
            )
        else()
            message(WARNING "Driver ${driver_name} unpacked to ${${target_name_lc}_SOURCE_DIR}, but no configuration method found.")
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
# is_unit_test      - Whether test will be run automatically
function(coda_add_tests_impl module_name dir_name deps filter_list is_unit_test)
    # Find all the source files, relative to the module's directory
    file(GLOB_RECURSE local_tests RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${dir_name}/*.cpp")
    # Filter out ignored files
    filter_files(local_tests "${local_tests}" "${filter_list}")

    # make a group target to build all tests for the current module
    set(test_group_tgt "${module_name}_tests")
    if (NOT TARGET ${test_group_tgt})
        add_custom_target(${test_group_tgt})
    endif()

    if (MSVC)
        add_compile_options(/W3) # change this to /W4 later
    endif()

    list(APPEND deps ${module_name})

    # we need the parent directory include for TestCase.h
    set(include_dirs "${CMAKE_CURRENT_SOURCE_DIR}/../${CODA_STD_PROJECT_INCLUDE_DIR}")

    # get all interface libraries and include directories from the dependencies
    foreach(dep ${deps})
        if (TARGET ${dep})
            get_property(dep_includes TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            list(APPEND include_dirs ${dep_includes})
        endif()
    endforeach()

    foreach(test_src ${local_tests})
        # Use the base name of the source file as the name of the test
        get_filename_component(test_name "${test_src}" NAME_WE)
        add_executable(${test_name} "${test_src}")
        add_dependencies(${test_group_tgt} ${test_name})
        get_filename_component(test_dir "${test_src}" DIRECTORY)
        # Do a bit of path manipulation to make sure tests in deeper subdirs retain those subdirs in their build outputs
#xxxTODO double-check this
        file(RELATIVE_PATH test_subdir "${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}" "${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}")
        # message(STATUS "Generating Test: module_name=${module_name} test_src=${test_src}  test_name=${test_name}  test_dir=${test_dir}  test_subdir= ${test_subdir} deps=${deps}")

        # Set IDE subfolder so that tests appear in their own tree
        set_target_properties(${test_name} PROPERTIES FOLDER "${dir_name}/${module_name}/${test_subdir}")

        target_link_libraries(${test_name} PRIVATE ${deps})
        target_include_directories(${test_name} PRIVATE ${include_dirs})

        # add unit tests to automatic test suite
        if (${is_unit_test})
            add_test(${test_name} ${test_name})
        endif()

        # Install [unit]tests to separate subtrees
        install(TARGETS ${test_name} RUNTIME DESTINATION "${dir_name}/${module_name}/${test_subdir}")
    endforeach()
endfunction()


function(coda_add_tests module_name deps filter_list)
    if (CODA_BUILD_TESTS)
        coda_add_tests_impl(${module_name} "${CODA_STD_PROJECT_TESTS_DIR}" "${deps}" "${filter_list}" FALSE)
    endif()
endfunction()

function(coda_add_unittests module_name deps filter_list)
    if (CODA_BUILD_TESTS)
        coda_add_tests_impl(${module_name} "${CODA_STD_PROJECT_UNITTESTS_DIR}" "${deps}" "${filter_list}" TRUE)
    endif()
endfunction()


# coda_add_library_impl() - Add a library to the build
#
# tgt_name          - Name of the module
# tgt_lang          - Language of the library
# tgt_deps          - List of linkable dependencies for the library
# tgt_extra_deps    - List of non-linkable dependencies for the library
# source_filter     - Source files to ignore
function(coda_add_library_impl tgt_name tgt_lang tgt_deps tgt_extra_deps source_filter)
    # Find all the source files, relative to the module's directory
    file(GLOB_RECURSE local_sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CODA_STD_PROJECT_SOURCE_DIR}/*.cpp")

    # Filter out ignored files
    filter_files(local_sources "${local_sources}" "${source_filter}")

    # Periods in target names for dirs are replaced with slashes (subdirectories).
    string(REPLACE "." "/" tgt_munged_dirname ${tgt_name})

    # Periods in target names for files are replaced with underscores.
    # Note that this variable name is used in the *.cmake.in files.
    string(REPLACE "." "_" tgt_munged_name ${tgt_name})

    # If we find a *_config.h.cmake.in file, generate the corresponding *_config.h, and put the
    #   target directory in the include path.
    #xxx This should probably look for all *.cmake.in files and process them.
    set(config_file_template "${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_munged_dirname}/${tgt_name}_config.h.cmake.in")
    if (EXISTS ${config_file_template})
        set(config_file_out "${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_munged_dirname}/${tgt_munged_name}_config.h")
        message(STATUS "Processing config header: ${config_file_template} -> ${config_file_out}")
        configure_file("${config_file_template}" "${config_file_out}")
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${config_file_out}" DESTINATION "${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_munged_dirname}")
    endif()

    if (NOT local_sources)
        # Libraries without sources must be declared to CMake as INTERFACE libraries
        set(header_type INTERFACE)
        add_library(${tgt_name} INTERFACE)
        if (tgt_deps)
            target_link_libraries(${tgt_name} INTERFACE ${tgt_deps})
        endif()
    else()
        set(header_type PUBLIC)
        add_library(${tgt_name} ${local_sources})
        if (tgt_lang)
            set_target_properties(${tgt_name} PROPERTIES OUTPUT_NAME "${tgt_name}-${tgt_lang}")
        endif()
        if (tgt_deps)
            target_link_libraries(${tgt_name} PUBLIC ${tgt_deps})
        endif()
    endif()

    # set our include directories
    set(include_dirs
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}>
        "${CMAKE_INSTALL_PREFIX}/${CODA_STD_PROJECT_INCLUDE_DIR}")
    # add interface include directories from the dependencies to our interface
    foreach (dep ${tgt_deps})
        if (TARGET ${dep})
            get_property(dep_include_dirs TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            list(APPEND include_dirs ${dep_include_dirs})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES include_dirs)
    target_include_directories(${tgt_name} ${header_type} ${include_dirs})

    if (tgt_extra_deps)
        add_dependencies(${tgt_name} ${tgt_extra_deps})
    endif()

    if (MSVC)
        add_compile_options(/W3) # change this to /W4 later
    endif()

    # Set up install destinations for binaries
    install(TARGETS ${tgt_name}
            EXPORT "${tgt_name}_TARGETS"
            LIBRARY DESTINATION "${CODA_STD_PROJECT_LIB_DIR}"
            ARCHIVE DESTINATION "${CODA_STD_PROJECT_LIB_DIR}"
            RUNTIME DESTINATION "${CODA_STD_PROJECT_BIN_DIR}")

    # Set up install destination for headers
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}"
            DESTINATION "."
            FILES_MATCHING
                PATTERN "*.h"
                PATTERN "*.hpp")

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
endfunction()


# coda_add_library_impl() - Add a SWIG Python module to the build
#
# tgt_name          - Name of the CMake target to build the module
# module_name       - Name of the module
# deps              - List of linkable dependencies for the library
# python_deps       - List of Python module dependencies for the library
# input_file        - Source file (.i) from which to generate the SWIG bindings
function(coda_add_swig_python_module_impl tgt_name module_name deps python_deps input_file)
    # determine all of the necessary include dirs and link libs from the dependencies
    set(include_dirs $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source>)
    set(libs "")
    foreach(dep ${deps})
        get_property(dep_interface_include_dirs TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(APPEND include_dirs ${dep_interface_include_dirs})

        list(APPEND libs ${dep})
        # get any transitive dependencies
        if (TARGET ${dep})
            get_property(dep_interface_link_libs TARGET ${dep} PROPERTY INTERFACE_LINK_LIBRARIES)
            list(APPEND libs ${dep_interface_link_libs})
        endif()
    endforeach()
    list(APPEND libs ${Python_LIBRARIES})

    # get SWIG include directories from the Python dependencies
    foreach(dep ${python_deps})
        get_property(dep_swig_include_dirs TARGET ${dep} PROPERTY SWIG_INCLUDE_DIRECTORIES)
        list(APPEND include_dirs ${dep_swig_include_dirs})
    endforeach()

    set_property(SOURCE ${input_file} PROPERTY CPLUSPLUS ON)
    set_property(SOURCE ${input_file} PROPERTY SWIG_MODULE_NAME ${module_name})
    set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/source/generated")
    set(SWIG_OUTFILE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/generated")

    swig_add_library(${tgt_name} LANGUAGE python SOURCES ${input_file})

    swig_link_libraries(${tgt_name} "${libs}")
    set_property(TARGET ${tgt_name} PROPERTY
        SWIG_INCLUDE_DIRECTORIES "${include_dirs}")
    set_property(TARGET ${tgt_name} PROPERTY
        SWIG_GENERATED_INCLUDE_DIRECTORIES "${Python_INCLUDE_DIRS}")
    set_property(TARGET ${tgt_name} PROPERTY
        LIBRARY_OUTPUT_NAME ${module_name})
    install(TARGETS ${tgt_name}
            DESTINATION "${CODA_PYTHON_SITE_PACKAGES}/coda")
endfunction()


# Add a SWIG Python module and its associated tests to the build.
#
#   This is a wrapper function to facilitate calls to the above routines from sub-projects.
#
#   To simplify things for callers that don't want to use many of the potential arguments,
#     this method does not take formal parameters.  Instead, callers should set any of the
#     following variables to define the library:
#
#       MODULE_NAME     Name of the module within Python
#       MODULE_DEPS     List of dependencies for the library
#       PYTHON_DEPS     List of Python module dependencies for the library
#       SWIG_INPUT_FILE Source file (.i) from which to generate the SWIG bindings
#
#  The caller can then simply call coda_add_library(target_name)
#
function(coda_add_swig_python_module tgt_name)
    coda_add_swig_python_module_impl("${tgt_name}" "${MODULE_NAME}"
                                     "${MODULE_DEPS}" "${PYTHON_DEPS}" "${SWIG_INPUT_FILE}")
    # TODO add tests
endfunction()
