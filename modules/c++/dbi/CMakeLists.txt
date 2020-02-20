set(TARGET_NAME "dbi")
set(MODULE_DEPS "io")
#xxx set(USELIB "SQL")

set(CODA_SQL_LAYER "PostgreSQL")  # "MYSQL", "PostgreSQL", or "ORACLE"

set("USE_${CODA_SQL_LAYER}" ON)

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


if (${CODA_SQL_LAYER}_FOUND) # Don't bother building if we've got no implementation.
    coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.
endif()
