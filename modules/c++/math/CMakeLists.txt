set("TARGET_NAME"		"math")
set("MAINTAINER"		"asylvest@users.sourceforge.net")
set("TARGET_VERSION"	"0.1")
set("MODULE_DEPS"		"except" "str" "sys" "types")
set("USELIB"			"MATH")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.