#include "pch.h"

#include <windows.h>
#include <combaseapi.h>
#undef interface

#include <import/sys.h>

#include "nitf_Test.h"

struct test_j2k_loading__ : public ::testing::Test {
	test_j2k_loading__() {
		// initialization code here
		//const std::string NITF_PLUGIN_PATH = R"(C:\Users\jdsmith\source\repos\nitro\x64\Debug\share\nitf\plugins)";
		sys::OS().setEnv("NITF_PLUGIN_PATH", nitf::Test::buildPluginsDir(), true /*overwrite*/);
	}

	void SetUp() {
		// code here will execute just before the test ensues 
	}

	void TearDown() {
		// code here will be called just after the test completes
		// ok to through exceptions from here if need be
	}

	~test_j2k_loading__() {
		// cleanup any pending stuff, but no exceptions allowed
	}

	test_j2k_loading__(const test_j2k_loading__&) = delete;
	test_j2k_loading__& operator=(const test_j2k_loading__&) = delete;

	// put in any custom data members that you need 
};

#define TEST_CASE(X) TEST_F(test_j2k_loading__, X)
#include "nitf/unittests/test_j2k_loading++.cpp"