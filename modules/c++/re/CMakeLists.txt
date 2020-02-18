set("TARGET_NAME"		"re")
set("MAINTAINER"		"adam.sylvester@mdaus.com matthew.wysocki@mdaus.com")
set("TARGET_VERSION"	"1.0")
set("MODULE_DEPS"		"sys")

# Enable to use std::regex instead of PCRE.
set("RE_ENABLE_STD_REGEX" 0)

if (NOT RE_ENABLE_STD_REGEX)
	set("MODULE_DEPS" "${MODULE_DEPS}" "pcre2-8")
	#set("PCRE2_STATIC" 1) # I don't think this is used anywhere
endif()

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

if (NOT RE_ENABLE_STD_REGEX)
	# Because the pcre2-x targets just hard-path their include files instead of setting include
	# directories, we don't get those directories automatically.  Manually add what we need here.
    target_include_directories("${TARGET_NAME}" PUBLIC
        $<BUILD_INTERFACE:${PCRE2_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>)
endif()

#[[xxx
distclean = lambda p: None

from waflib import Options
from build import writeConfig

def options(opt):
    # We don't make this the default since unfortunately the STL implementation
    # is significantly slower than PCRE
    opt.add_option('--enable-std-regex', action='store_true', dest='enable_std_regex',
                   default=False,
                   help='turn on using STL\'s regex implementation.  Only works when C++11 support is enabled')

def configure(conf):
    conf.env['enable_std_regex'] = Options.options.enable_std_regex

    def re_callback(conf):
        conf.env['DEFINES'] = []
        if conf.env['enable_std_regex']:
            conf.define('RE_ENABLE_STD_REGEX', 1)
        elif not Options.options.shared_libs:
            # We need this in order to link properly on Windows
            # We could get this via export_defines in the modules/drivers/pcre/wscript
            # but that won't work if someone is trying to link in our stuff from their
            # own build system (or if they're using a prebuilt PCRE)
            conf.define('PCRE2_STATIC', 1)

    writeConfig(conf, re_callback, NAME)

def build(bld):
    # We only depend on PCRE if we're not using the STL regex
    modArgs = globals()
    if not bld.env['enable_std_regex']:
        modArgs['USELIB_CHECK'] = 'PCRE'

    bld.module(**modArgs)
]]
