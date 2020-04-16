from conans import ConanFile, CMake, tools

class CodaOssConan(ConanFile):
    name = "coda-oss"
    url = "https://github.com/mdaus/coda-oss"
    description = "Common Open Development Archive - OSS"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False],
               "CODA_PARTIAL_INSTALL": [True, False, None],
               "BOOST_HOME": "ANY",
               "PYTHON_HOME": "ANY",
               "ENABLE_J2K": [True, False, None],
               "J2K_HOME": "ANY",
               "ENABLE_JPEG": [True, False, None],
               "JPEG_HOME": "ANY",
               "ENABLE_PCRE": [True, False, None],
               "PCRE_HOME": "ANY",
               "SSL_HOME": "ANY",
               "UUID_HOME": "ANY",
               "XML_HOME": "ANY",
               "ZIP_HOME": "ANY",
               "MT_DEFAULT_PINNING": [True, False, None],
               }
    default_options = {"shared": False,
                       "BOOST_HOME": "",
                       "PYTHON_HOME": "",
                       "J2K_HOME": "",
                       "JPEG_HOME": "",
                       "PCRE_HOME": "",
                       "SSL_HOME": "",
                       "UUID_HOME": "",
                       "XML_HOME": "",
                       "ZIP_HOME": "",
                       }
    exports_sources = ("*.txt",
                       "*.md",
                       "LICENSE",
                       "build/*",
                       "cmake/*",
                       "modules/*",
                       )
    license = "GNU LESSER GENERAL PUBLIC LICENSE Version 3"

    def set_version(self):
        git = tools.Git(folder=self.recipe_folder)
        self.version = "%s_%s" % (git.get_branch(), git.get_revision()[:16])

    def _configure_cmake(self):
        cmake = CMake(self)
        # automatically foward all uppercase arguments to CMake
        for name, val in self.options.iteritems():
            if name.isupper() and val is not None:
                cmake.definitions[name] = val
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()

    def package_id(self):
        # Make any change in our dependencies' version require a new binary
        self.info.requires.full_version_mode()

    def package_info(self):
        self.cpp_info.builddirs = ["lib/cmake"]
