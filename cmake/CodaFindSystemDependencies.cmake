# find and import system dependencies
macro(coda_find_system_dependencies)
    # creates imported target Threads::Threads
    # see https://cmake.org/cmake/help/latest/module/FindThreads.html
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads)

    # creates imported target CURL::libcurl, if found
    # see https://cmake.org/cmake/help/latest/module/FindCURL.html
    find_package(CURL)
    if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
        #FindCurl didn't create a target until CMake 3.12
        if(CURL_FOUND)
            if(NOT TARGET CURL::libcurl)
                add_library(CURL::libcurl UNKNOWN IMPORTED)
                set_target_properties(CURL::libcurl
                    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CURL_INCLUDE_DIRS}")
                set_property(TARGET CURL::libcurl
                    APPEND PROPERTY IMPORTED_LOCATION "${CURL_LIBRARY}")
            endif()
        endif()
    endif()

    # creates imported target Boost::serialization, if found
    # see https://cmake.org/cmake/help/latest/module/FindBoost.html
    set(BOOST_HOME "" CACHE PATH "path to boost installation")
    if (BOOST_HOME)
        set(BOOST_ROOT ${BOOST_HOME})
    endif()
    find_package(Boost COMPONENTS serialization)
    set(HAVE_BOOST ${Boost_FOUND})

    # sets the following variables if Python installation found:
    #   Python_FOUND                - flag indicating system has the requested components
    #   Python_VERSION
    #   Python_VERSION_MAJOR
    #   Python_VERSION_MINOR
    #   Python_VERSION_PATCH
    #
    #   Python_Interpreter_FOUND    - flag indicating system has the interpreter
    #   Python_EXECUTABLE           - path to interpreter
    #   Python_STDLIB               - standard installation directory
    #
    #   Python_Development_FOUND    - flag indicating system has development artifacts
    #   Python_INCLUDE_DIRS         - Python include directories
    #   Python_LIBRARIES            - Python libraries
    #
    #   Python_NumPy_FOUND          - flag indicating system has NumPy
    #   Python_NumPy_INCLUDE_DIRS   - NumPy include directories
    #
    # see https://cmake.org/cmake/help/latest/module/FindPython.html
    if (PYTHON_HOME)
        set(Python_ROOT_DIR ${PYTHON_HOME})
    endif()
    find_package(Python COMPONENTS Interpreter Development NumPy)
    if (Python_FOUND AND Python_Development_FOUND)
        set(CODA_PYTHON_SITE_PACKAGES
            "${CODA_STD_PROJECT_LIB_DIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
        if(NOT PYTHON_HOME)
            message("Python installation found at ${Python_EXECUTABLE}.\n"
                    "Pass the configure option -DPYTHON_HOME=... to override this selection.")
        endif()
        if (Python_Interpreter_FOUND)
            message("Python_EXECUTABLE=${Python_EXECUTABLE}")
        endif()
        if (Python_Development_FOUND)
            message("Python_INCLUDE_DIRS=${Python_INCLUDE_DIRS}")
            message("Python_LIBRARIES=${Python_LIBRARIES}")
        endif()
        if (Python_NumPy_FOUND)
            message("Python_NumPy_INCLUDE_DIRS=${Python_NumPy_INCLUDE_DIRS}")
        endif()
    else()
        message(WARNING "Python targets will not be built since Python libraries were not found.\n"
                "Pass the configure option -DPYTHON_HOME=... to help locate an installation.")
    endif()

    # if SWIG found, sets path to swig utilities in SWIG_USE_FILE
    # see https://cmake.org/cmake/help/latest/module/UseSWIG.html
    find_package(SWIG)
    if (SWIG_FOUND)
        if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
            cmake_policy(SET CMP0078 NEW) # UseSWIG generates standard target names
        endif()
        if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14")
            cmake_policy(SET CMP0086 NEW) # UseSWIG honors SWIG_MODULE_NAME via -module
        endif()
        include(${SWIG_USE_FILE})
        option(PYTHON_EXTRA_NATIVE "generate extra native containers with SWIG" ON)
        if (PYTHON_EXTRA_NATIVE)
            list(APPEND CMAKE_SWIG_FLAGS "-extranative")
        endif()
        if (Python_VERSION_MAJOR GREATER_EQUAL 3)
            list(APPEND CMAKE_SWIG_FLAGS "-py3")
        endif()
    else()
        message(WARNING "SWIG could not be found, so modules relying on it will not be built")
    endif()

    # sets OPENSSL_FOUND to TRUE if found, and creates targets
    # OpenSSL:SSL and OpenSSL::Crypto
    # see https://cmake.org/cmake/help/latest/module/FindOpenSSL.html
    find_package(OpenSSL)
endmacro()
