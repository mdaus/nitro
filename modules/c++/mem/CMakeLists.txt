set("TARGET_NAME"		"mem")
set("MAINTAINER"		"asylvest@users.sourceforge.net")
set("TARGET_VERSION"	"1.0")
set("MODULE_DEPS"		"sys")
set("UNITTEST_DEPS"		"sys")  #xxx Ideally this would be implicit.

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.