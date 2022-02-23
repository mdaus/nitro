#include "pch.h"

#include <windows.h>
#include <combaseapi.h>
#undef interface

#include "nitf_Test.h"

#define TEST_CASE(X) TEST(test_j2k_read_tile, X)
#include "nitf/unittests/test_j2k_read_tile.cpp"
