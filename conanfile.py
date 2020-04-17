from conans import ConanFile, CMake, tools

class CodaOssConan(ConanFile):
    name = "coda-oss"
    url = "https://github.com/mdaus/coda-oss"
    description = "Common Open Development Archive - OSS"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False],
               "BOOST_HOME": "ANY",
               "MT_DEFAULT_PINNING": [True, False, None],
               }
    default_options = {"shared": False,
                       "BOOST_HOME": "",
                       "MT_DEFAULT_PINNING": None, # None means use CMake default
                       }
    exports_sources = ("CMakeLists.txt",
                       "LICENSE",
                       "README.md",
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
