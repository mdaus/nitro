#include "pch.h"

#include <windows.h>
#include <combaseapi.h>
#undef interface

#include <import/sys.h>

#include "nitf_Test.h"

#define TEST_CASE(X) TEST(test_j2k_loading__, X)
#include "nitf/unittests/test_j2k_loading++.cpp"