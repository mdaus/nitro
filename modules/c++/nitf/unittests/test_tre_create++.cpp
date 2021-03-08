
#include <import/nitf.hpp>

#include "TestCase.h"

TEST_CASE(test_nitf_create_tre)
{
    // https://github.com/mdaus/nitro/issues/329

    nitf::TRE tre("HISTOA", "HISTOA"); // allocates fields SYSTEM .. NEVENTS
    tre.setField("SYSTYPE", "M1");
    TEST_ASSERT_TRUE(true);

    tre.setField("PDATE[0]", "20200210184728"); // aborts program; worked prior to 2.10
    TEST_ASSERT_TRUE(true);
}

TEST_MAIN(
    (void)argc;
argv0 = argv[0];

TEST_CHECK(test_nitf_create_tre);
)