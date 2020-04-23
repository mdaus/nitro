# CMake build system for CODA-oss #

August 2018, Scott.Colcord@maxar.com <br/>
April 2020, Keith.Wilhelm@maxar.com

CODA-oss now contains a CMake-based build system.

### Dependencies ###
#### Required ####
* CMake >= 3.14, configured with "--qt-gui" if the qt gui is desired.
#### Optional ####
* Python (help locate by passing `-DPYTHON_HOME=[path]`)
* SWIG (help locate by adding binary to the PATH)
* Boost (help locate by passing `-DBOOST_HOME=[path]`)
* curl


### Files ###
* CMakeLists.txt - Entry point for the CMake build.
* cmake/CodaBuild.cmake - CMake build tools
* cmake/FindSystemDependencies.cmake - Tools for finding system dependencies, needed by consumers
* modules/.../CMakeLists.txt - Project specific options and settings.
* modules/.../include/.../*.cmake.in - Templates for generating config header files.

## How to build ##
### Instructions for Linux ###
```
# starting from base directory of repo, make a build directory and cd into it
mkdir target
cd target

# configure
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=[install path]

# build and install, in parallel
cmake --build . --target install -j

# run unit tests (optional)
ctest
```

### Instructions for Windows ###
(same as above, but build type must be specified
during the build step instead of the configure step):
```
# starting from base directory of repo, make a build directory and cd into it
mkdir target
cd target

# configure
cmake .. -DCMAKE_INSTALL_PREFIX=[install path]

# build and install, in parallel
cmake --build . --config Release --target install -j

# run unit tests (optional)
ctest -C Release
```

Or open the directory containing the top-level CMakeLists.txt as a project in Visual Studio.

## Build configuration ##
These are additional configuration options which may be passed in the cmake configure step as `-DOPTION_NAME=[option value]`
| Setting        | Default | Description |
|----------------|---------|-------------|
|CMAKE_BUILD_TYPE|RelWithDebInfo|build type (Release, Debug, RelWithDebInfo); not used for MSVC builds (should be specified at build time instead)|
|BUILD_SHARED_LIBS|OFF|build shared libraries if on, static if off|
|CODA_BUILD_TESTS| ON      |variable in the script controls whether tests will be built|
|BOOST_HOME||path to existing Boost installation|
|ENABLE_PYTHON|ON|build Python modules if enabled|
|PYTHON_HOME||path to existing Python installation (implies ENABLE_PYTHON=ON)|
|ENABLE_JPEG|ON|build libjpeg driver and modules depending on it|
|JPEG_HOME||path to existing libjpeg installation; if not provided, it will be built from source (implies ENABLE_JPEG=ON)|
|ENABLE_J2K|ON|build openjpeg (jpeg2000) driver and modules depending on it|
|J2K_HOME||path to existing openjpeg installation; if not provided, it will be built from source (implies ENABLE_J2K=ON)|
|ENABLE_PCRE|ON|build PCRE (PERL Compatible Regular Expressions) library and modules dependent on it|
|PCRE_HOME||path to existing pcre installation; if not provided, it will be built from source (implies ENABLE_PCRE=ON)|
|XML_HOME||path to existing Xerces installation; if not provided, it will be built from source|
|ZIP_HOME||path to existing zlib installation; if not provided, it will be built from source|


## Issues ##
* Shared library build needs testing.

