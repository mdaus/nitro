#include "pch.h"
#include "CppUnitTest.h"

#include <nitf/PluginRegistry.h>
#include <nitf/UnitTests.hpp>

// https://learn.microsoft.com/en-us/visualstudio/test/microsoft-visualstudio-testtools-cppunittestframework-api-reference?view=vs-2022
TEST_MODULE_INITIALIZE(methodName)
{
    nitf_PluginRegistry_PreloadedTREHandlersEnable(NRT_TRUE);

    // module initialization code
    nitf::Test::j2kSetNitfPluginPath();
}