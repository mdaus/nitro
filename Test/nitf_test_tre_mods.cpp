#include "pch.h"

#include <string>

#include "nitf_Test.h"



struct nitf_test_tre_mods : public ::testing::Test {
    nitf_test_tre_mods() {
        // initialization code here
        const std::string NITF_PLUGIN_PATH = R"(C:\Users\jdsmith\source\repos\nitro\x64\Debug\share\nitf\plugins)";
        const std::string putenv_ = "NITF_PLUGIN_PATH=" + NITF_PLUGIN_PATH;

        _putenv(putenv_.c_str());
    }

    void SetUp() {
        // code here will execute just before the test ensues 
    }

    void TearDown() {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~nitf_test_tre_mods() {
        // cleanup any pending stuff, but no exceptions allowed
    }

    // put in any custom data members that you need 
};

#define TEST_CASE(X) TEST_F(nitf_test_tre_mods, X)
#include "nitf/unittests/test_tre_mods.c"
#include "nitf/unittests/test_tre_mods++.cpp"

