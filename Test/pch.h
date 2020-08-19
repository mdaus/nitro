//
// pch.h
// Header for standard system include files.
//

#pragma once

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <ctype.h>
#include <time.h>
#include <stdint.h>

#include <string>
#include <ios>

#include <windows.h>
#undef min
#undef max
#  include <direct.h>
#  include <io.h>
# include <sys/types.h>
# include <sys/stat.h>

extern int close(int fd);
extern int read(int fd, void* buf, unsigned int count);
extern int write(int fd, const void* buf, unsigned int count);
extern int _isatty(int fd);
#include "gtest/gtest.h"

#include <import/nrt.h>

#include "nitf_Test.h"
#include "Test.h"

#pragma comment(lib, "mt-c++")
#pragma comment(lib, "io-c++")
#pragma comment(lib, "except-c++")
#pragma comment(lib, "sys-c++")
#pragma comment(lib, "str-c++")

#pragma comment(lib, "ws2_32")
