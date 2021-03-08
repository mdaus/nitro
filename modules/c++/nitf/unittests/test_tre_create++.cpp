#include <stdlib.h>

#include <import/nitf.hpp>

#include "TestCase.h"

TEST_CASE(test_tre_create_329)
{
    // https://github.com/mdaus/nitro/issues/329

    nitf::TRE tre("HISTOA", "HISTOA"); // allocates fields SYSTEM .. NEVENTS
    tre.setField("SYSTYPE", "M1");
    TEST_ASSERT_TRUE(true);

    //tre.setField("PDATE[0]", "20200210184728"); // aborts program; worked prior to 2.10
    TEST_ASSERT_TRUE(true);
}

TEST_CASE(test_tre_clone_329)
{
    // https://github.com/mdaus/nitro/issues/329
    const std::string rd = "begin1020030004ABCDEFend";

    nitf_Error error;
    auto rawTre = nitf_TRE_construct("TESTxyz", NITF_TRE_RAW, &error);
    nitf_TRE_setField(rawTre, "raw_data", const_cast<char*>(rd.c_str()), rd.size(), &error);
    TEST_ASSERT_TRUE(true);

    nitf::TRE cloneTre(rawTre); // leads to free(): invalid size(): in libc.so.6 when the program exits; worked prior to 2.10
    TEST_ASSERT_TRUE(true);
}

int main()
{
    TEST_CHECK(test_tre_create_329);
    TEST_CHECK(test_tre_clone_329);

    return EXIT_SUCCESS;
}
