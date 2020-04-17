from conans import ConanFile, CMake, tools

class NitroConan(ConanFile):
    name = "nitro"
    url = "https://github.com/mdaus/nitro"
    description = "library for reading and writing the National Imagery Transmission Format (NITF)"
    settings = "os", "compiler", "build_type", "arch"
    requires = ("coda-oss/CMake_update_win_12c8a1574ed4694c@user/testing", )
    options = {"shared": [True, False],
               }
    default_options = {"shared": False,
                       }
    exports_sources = ("CMakeLists.txt",
                       "LICENSE",
                       "README.md",
                       "cmake/*",
                       "modules/*",
                       )
    generators = "cmake_paths"
    license = "GNU LESSER GENERAL PUBLIC LICENSE Version 3"

    def set_version(self):
        git = tools.Git(folder=self.recipe_folder)
        self.version = "%s_%s" % (git.get_branch(), git.get_revision()[:16])

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["ENABLE_STATIC_TRES"] = True # always build static TRES
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
