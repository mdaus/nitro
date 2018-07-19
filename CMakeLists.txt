# - CODA-OSS CMake Build script.
#
# Author:  Scott A. Colcord

cmake_minimum_required(VERSION 3.4)

project("coda-oss")

# Standard directory names used throughout the project.
#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR/target}) # Specified in CMakeSettings.json
set(CODA_STD_BUILD_DIR				"build")
set(CODA_STD_PROJECT_SOURCE_DIR		"source")
set(CODA_STD_PROJECT_INCLUDE_DIR	"include")
set(CODA_STD_PROJECT_IMPORT_DIR		"import")
set(CODA_STD_PROJECT_LIB_DIR		"lib")
set(CODA_STD_PROJECT_BIN_DIR		"bin")
set(CODA_STD_PROJECT_TESTS_DIR		"tests")
set(CODA_STD_PROJECT_UNITTESTS_DIR	"unittests")


include(${CODA_STD_BUILD_DIR}/config_tests.cmake) # Code to test compiler features

#set(BUILD_SHARED_LIBS OFF)  # Currently done via an option passed to CMake

enable_testing()


# MSVC-specific flags and options.
if (MSVC)
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)

	# Remove any default settings that we don't want.
	string(REGEX REPLACE "/W[0-3]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	string(REGEX REPLACE "/W[0-3]" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
	string(REGEX REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

	add_definitions(
		-DWIN32_LEAN_AND_MEAN 
		-DNOMINMAX
		-D_CRT_SECURE_NO_WARNINGS
		-D_SCL_SECURE_NO_WARNINGS
		-D_USE_MATH_DEFINES
	)
	add_compile_options(
		/W3					#xxx Change this to /W4 later
		/wd4290
		/wd4512
		/EHs				# Needed when exceptions might bubble through a C-linkage layer
#		/MP					# Parallel Build
	)

	link_libraries(			# CMake uses this for both libraries and linker options
		-STACK:80000000
	)

	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
	set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD TRUE)
endif()

# Unix/Linux specific options
if (UNIX)
	add_definitions(
		-D_LARGEFILE_SOURCE
		-D_FILE_OFFSET_BITS=64
	)
endif()

# filter_files()- Utility to filter a list of files.
#
# dest_name		- Destination variable name in parent's scope
# file_list		- Input list of files (possibly with paths)
# filter_list	- Input list of files to filter out 
#					(must be bare filenames; no paths)
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

# coda_add_tests()	- Add a module's tests or unit tests to the build
#
# module_name		- Name of the module
# dir_name			- Subdirectory containing the tests' source code
#					  All source files beneath this directory will be used.
#                     Each source file is assumed to create a separate executable.
# deps				- Modules that the tests are dependent upon.
# filter_list		- Source files to ignore
function(coda_add_tests module_name dir_name deps filter_list)
	# Find all the source files, relative to the module's directory
	file(GLOB_RECURSE local_tests RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${dir_name}/*.cpp")
	# Filter out ignored files
	filter_files(local_tests "${local_tests}" "${filter_list}")

	foreach(test_src ${local_tests})
		# Use the base name of the source file as the name of the test
		get_filename_component(test_name ${test_src} NAME_WE) 
		add_executable(${test_name} "${test_src}")
		get_filename_component(test_dir ${test_src} DIRECTORY)
		# Do a bit of path manipulation to make sure tests in deeper subdirs retain those subdirs in their build outputs
#xxxTODO double-check this
		file(RELATIVE_PATH test_subdir "${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}" "${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}")
		#message("test_src=${test_src}  test_name=${test_name}  test_dir=${test_dir}  test_subdir= ${test_subdir}")
		# Set IDE subfolder so that tests appear in their own tree
		set_target_properties(${test_name} PROPERTIES FOLDER "${dir_name}/${module_name}/${test_subdir}")
		# We need the parent directory include for TestCase.h
		target_include_directories(${test_name}  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../${CODA_STD_PROJECT_INCLUDE_DIR}")
#xxx This shouldn't be needed; it should come from the module.
#		target_include_directories(${test_name}  PUBLIC "${CODA_STD_PROJECT_INCLUDE_DIR}")
		# Automatically depend on the parent module, plus any others that were specified
		target_link_libraries(${test_name} ${module_name} ${deps})

		# Install [unit]tests to separate subtrees
		install(TARGETS ${test_name} RUNTIME DESTINATION "${dir_name}/${module_name}/${test_subdir}")
	endforeach()
endfunction()

# coda_add_library_impl() - Add a library to the build
#
# module_name		- Name of the module
# dir_name			- Subdirectory containing the tests' source code
#					  All source files beneath this directory will be used.
#                     Each source file is assumed to create a separate executable.
# deps				- Modules that the tests are dependent upon.
# filter_list		- Source files to ignore
function(coda_add_library_impl tgt_name tgt_lang tgt_version tgt_deps source_filter)

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

	add_library("${tgt_name}" ${lib_type} ${local_sources})

	# Find all the header files, relative to the module's directory, and add them.
#	file(GLOB_RECURSE local_headers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_name}/*.h")
	file(GLOB_RECURSE local_headers "${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_name}/*.h")
#	message("tgt_name=${tgt_name}  local_headers=${local_headers}")
	target_sources("${tgt_name}" "${header_type}" ${local_headers})

	# For non-INTERFACE libraries, assign a version number and an appropriate output name
	#   and add dependencies.
	if (NOT lib_type STREQUAL "INTERFACE")
		set_target_properties("${tgt_name}" PROPERTIES 
			OUTPUT_NAME "${tgt_name}-${tgt_lang}"
			VERSION "${tgt_version}"
		)
		target_link_libraries("${tgt_name}" ${tgt_deps})
	endif()

	# Add include directories
	file(GLOB local_header_dirs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CODA_STD_PROJECT_INCLUDE_DIR}/*")
	target_include_directories("${tgt_name}" ${header_type} "${CODA_STD_PROJECT_INCLUDE_DIR}")

	# If we find a *_config.h.cmake.in file, generate the corresponding *_config.h, and put the 
	#   target directory in the include path.
	#xxx This should probably look for all *.cmake.in files and process them.
	set(config_filename "${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_name}/${tgt_name}_config.h.cmake.in")
	if (EXISTS "${config_filename}")
		configure_file("${config_filename}" "${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_name}/${tgt_name}_config.h")
		target_include_directories("${tgt_name}" PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}")
#xxx This is intended to put the generated header in the IDE file list, but it doesn't seem to work.
#		target_sources("${tgt_name}" "${header_type}" "${CMAKE_CURRENT_BINARY_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_name}/${tgt_name}_config.h")
	endif()

	# Set up install destinations for binaries
	install(TARGETS "${tgt_name}"
#xxx		EXPORT "$tgt_name_TARGETS"
		LIBRARY DESTINATION ${CODA_STD_PROJECT_LIB_DIR}
		ARCHIVE DESTINATION ${CODA_STD_PROJECT_LIB_DIR}
		RUNTIME DESTINATION ${CODA_STD_PROJECT_BIN_DIR}
	)
	# Set up install destination for headers
	install(DIRECTORY ${local_header_dirs} DESTINATION ${CODA_STD_PROJECT_INCLUDE_DIR})

	#xxx? Perhaps change the above to:
	#install(INCLUDES DESTINATION ${CODA_STD_PROJECT_INCLUDE_DIR})

#[[  #xxx Export the library interface?
	install(EXPORT "${tgt_name}_TARGETS"
		FILE ${tgt_name}_TARGETS.cmake
		NAMESPACE ${tgt_name}::
		DESTINATION ${CODA_STD_PROJECT_LIB_DIR}/cmake/${tgt_name}
	)

	include(CMakePckageConfigHelpers)
	write_basic_package_version_file(
		VERSION ${${tgt_name}_VERSION}
		COMPATIBILITY SameMajorVersion
	)
	install(FILES "${tgt_name}_config.cmake" "${tgt_name}_config_version.cmake"
		DESTINATION ${CODA_STD_PROJECT_LIB_DIR/cmake/${tgt_name}
	)
	# Then, pull in:
	include(CMakeFindDependencyMacro)
	find_dependency(mydepend 2.1)
	include("${CMAKE_CURRENT_LIST_DIR}/${tgt_name}_TARGETS.cmake")

	#xxx Note that it will probably also necessary to add the $<INSTALL_INTERFACE:include> to the target include dirs.

	#xxx Also, add_library("${tgt_name}::${tgt_name}" ALIAS ${tgt_name})
]]
endfunction()

# Add a library and its associated tests to the build.
# 
#   This is a wrapper function to facilitate calls to the above routines from sub-projects.
#
#   To simplify things for callers that don't want to use many of the potential arguments,
#     this method does not take formal parameters.  Instead, callers should set any of the
#     following variables to define the library:
#
#		TARGET_LANG		Language of the library
#		TARGET_VERSION	Version number of the library
#		MODULE_DEPS		List of dependencies for the library
#		SOURCE_FILTER	Source files to ignore
#
#	Directories defined by the variables CODA_STD_PROJECT_TESTS_DIR and 
#	  CODA_STD_PROJECT_UNITTESTS_DIR will be searched for source files; each of these
#	  will be compiled into test executables. The following variables affect the test
#	  executable creation:
#
#		TEST_DEPS		- List of dependencies for the files under CODA_STD_PROJECT_TESTS_DIR
#		TEST_FILTER		- List of source files to ignore under CODA_STD_PROJECT_TESTS_DIR
#		UNITTEST_DEPS	- List of dependencies for the files under CODA_STD_PROJECT_UNITTESTS_DIR
#		UNITTEST_FILTER	- List of source files to ignore under CODA_STD_PROJECT_UNITTESTS_DIR
# 
#  The caller can then simply call coda_add_library
#
function(coda_add_library tgt_name)
	coda_add_library_impl("${tgt_name}" "${TARGET_LANG}" "${TARGET_VERSION}" "${MODULE_DEPS}" "${SOURCE_FILTER}")
	coda_add_tests("${tgt_name}" "${CODA_STD_PROJECT_TESTS_DIR}" "${TEST_DEPS}" "${TEST_FILTER}")
	coda_add_tests("${tgt_name}" "${CODA_STD_PROJECT_UNITTESTS_DIR}" "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")
endfunction()

add_subdirectory("modules/c++")
#add_subdirectory("modules/python")

#add_subdirectory("modules/drivers/boost")
#add_subdirectory("modules/drivers/curl")
#add_subdirectory("modules/drivers/j2k")
add_subdirectory("modules/drivers/jars")
#add_subdirectory("modules/drivers/jpeg")
#add_subdirectory("modules/drivers/numpy")
#add_subdirectory("modules/drivers/openssl")
#add_subdirectory("modules/drivers/pcre")
#add_subdirectory("modules/drivers/sql")
#add_subdirectory("modules/drivers/uuid")
#add_subdirectory("modules/drivers/xml")
#add_subdirectory("modules/drivers/zlib")