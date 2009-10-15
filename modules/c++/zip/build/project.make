PACKAGE = zip
BUILD = library unit-tests
MAJOR_VERSION = 0
MINOR_VERSION = 1
VERSION_SUFFIX = dev
VERSION = $(MAJOR_VERSION)_$(MINOR_VERSION)$(VERSION_SUFFIX)
MAINTAINER = Dan.Pressel@gd-ais.com, Thomas.Zellman@gd-ais.com
DESCRIPTION = Compression/Decompression IO interface to zlib

SOURCES = GZipInputStream.cpp GZipOutputStream.cpp

TESTS = test_compress.cpp test_decompress.cpp

MY_CXXFLAGS = 
MY_CXXDEFINES =
MY_LFLAGS =
MY_INCLUDES = -I../../except/include \
              -I../../str/include \
              -I../../sys/include \
              -I../../io/include
MY_LIBPATH = -L../../except/lib/i686-pc-linux-gnu/gnu \
             -L../../str/lib/i686-pc-linux-gnu/gnu \
             -L../../sys/lib/i686-pc-linux-gnu/gnu \
             -L../../io/lib/i686-pc-linux-gnu/gnu
MY_LIBS = -lio-c++ -lsys-c++ -lstr-c++ -lexcept-c++ -lz
