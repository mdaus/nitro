#include "pch.h"

#include <import/sys.h>

#include "nitf_Test.h"

struct nitf_test_tre_create__ : public ::testing::Test {
	nitf_test_tre_create__() {
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

	~nitf_test_tre_create__() {
		// cleanup any pending stuff, but no exceptions allowed
	}

	// put in any custom data members that you need 
};

#define TEST_CASE(X) TEST_F(nitf_test_tre_create__, X)
#include "nitf/unittests/test_tre_create++.cpp"
