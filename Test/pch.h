//
// pch.h
// Header for standard system include files.
//
#pragma once

#pragma warning(disable: 4820) // '...': '...' bytes padding added after data member '...'
#pragma warning(disable: 4710) // '...': function not inlined
#pragma warning(disable: 5045) // Compiler will insert Spectre mitigation for memory load if / Qspectre switch specified
#pragma warning(disable: 4668) // '...' is not defined as a preprocessor macro, replacing with '...' for '...'
// TODO: get rid of these someday?
#pragma warning(disable: 5039) //	'...': pointer or reference to potentially throwing function passed to 'extern "C"' function under -EHc. Undefined behavior may occur if this function throws an exception.
#pragma warning(disable: 4514) //	'...': unreferenced inline function has been removed

#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#include "../modules/c++/cpp.h"
#pragma warning(pop)
#pragma comment(lib, "ws2_32")

// We're building in Visual Studio ... used to control where we get a little bit of config info
#define NITRO_PCH 1

#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#include <nitf/coda-oss.hpp>
#pragma comment(lib, "io-c++")
#pragma comment(lib, "except-c++")
#pragma comment(lib, "sys-c++")
#pragma comment(lib, "str-c++")
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable: 4388) // '...': signed / unsigned mismatch
#pragma warning(disable: 4389) // '...': signed / unsigned mismatch
#pragma warning(disable: 4800) // Implicit conversion from '...' to bool. Possible information loss
#pragma warning(disable: 4625) // '...': copy constructor was implicitly defined as deleted
#pragma warning(disable: 4626) // '...': assignment operator was implicitly defined as deleted
#pragma warning(disable: 5026) // '...': move constructor was implicitly defined as deleted
#pragma warning(disable: 5027) //	'...': move assignment operator was implicitly defined as deleted
#include "gtest/gtest.h"
#pragma warning(pop)

#include <import/nrt.h>
#include <nitf/System.hpp>

#include "nitf_Test.h"
#include "Test.h"


