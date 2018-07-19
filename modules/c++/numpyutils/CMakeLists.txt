set("TARGET_NAME"		"numpyutils")
set("MAINTAINER"		"anuraag.pakanati@mdaus.com")
set("TARGET_VERSION"	"1.0")
set("MODULE_DEPS"		"types" "except" "sys")
#xxx set("USELIB"			"NUMPY" "PYEXT")
coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

#[[xxx
from waflib import Options

def configure(conf):
    conf.recurse(['examples'])

def build(bld):
    if 'PY_HAVE_NUMPY' in bld.env and bld.env['PY_HAVE_NUMPY']:
       bld.module(**globals())
       bld.recurse(['examples'])
]]