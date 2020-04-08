set(MODULE_NAME dbi)

set(CODA_SQL_LAYER "PostgreSQL" CACHE STRING
    "the database to use, one of [PostgreSQL, MYSQL, ORACLE]")

set("USE_${CODA_SQL_LAYER}" ON CACHE INTERNAL "whether this SQL layer is used")

# Find the desired XML package and add its imported target to our dependencies
include("Find${CODA_SQL_LAYER}")
find_package(${CODA_SQL_LAYER})
if (${CODA_SQL_LAYER}_FOUND)
    list(APPEND "MODULE_DEPS" "${CODA_SQL_LAYER}::${CODA_SQL_LAYER}")
endif()

# The imported target wasn't added to FindPostgreSQL until CMake 3.12
if (USE_PostgreSQL)
    if(PostgreSQL AND NOT TARGET PostgreSQL::PostgreSQL)
        add_library(PostgreSQL::PostgreSQL UNKNOWN IMPORTED)
        set_target_properties(PostgreSQL::PostgreSQL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PostgreSQL_INCLUDE_DIRS}")
        set_property(TARGET PostgreSQL::PostgreSQL APPEND PROPERTY IMPORTED_LOCATION "${PostgrSQL_LIBRARY}")
    endif()
endif()

coda_add_module(
    NAME ${MODULE_NAME}
    VERSION 1.0
    DEPS io)  # See the definition of this method for details on the variables it looks at.

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")
