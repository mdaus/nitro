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

#include <windows.h>
#undef min
#undef max

#include "gtest/gtest.h"

#include <import/nrt.h>

#include "nitf_Test.h"
#include "Test.h"

#pragma comment(lib, "str-c++")
#pragma comment(lib, "mt-c++")
#pragma comment(lib, "except-c++")
#pragma comment(lib, "sys-c++")

#pragma comment(lib, "ws2_32")
