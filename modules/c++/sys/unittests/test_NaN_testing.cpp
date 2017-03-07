#include "TestCase.h"
#include <limits>

namespace
{
TEST_CASE(testNaNsAreNotEqual)
{
    // This test exists mainly to document behavior
    // It's not awesome that things work this way, but presumably
    // the caller is testing against a known value, so if this comes up
    // NaN oddness is already expected.
    TEST_ASSERT_NOT_EQ(std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN());

    TEST_ASSERT_NOT_EQ(std::numeric_limits<float>::quiet_NaN(), 3.4);

}
TEST_CASE(testNaNIsNotAlmostEqualToNumber)
{
    // Uncomment the test to see it work.
    // These test macros are supposed to fail here.
    // But I don't have a way to intercept the failure.
    // the failure.
    //
    /*
    TEST_ASSERT_ALMOST_EQ(std::numeric_limits<float>::quiet_NaN(), 5);
    TEST_ASSERT_ALMOST_EQ_EPS(std::numeric_limits<float>::quiet_NaN(),
                5, 3);
    */
}

TEST_CASE(testIsNaN)
{
    TEST_ASSERT_TRUE(IS_NAN(std::numeric_limits<float>::quiet_NaN()));
    TEST_ASSERT_FALSE(IS_NAN(5));
    TEST_ASSERT_FALSE(IS_NAN(std::string("test string")));
}
}

int main(int /*argc*/, char** /*argv*/)
{
    TEST_CHECK(testNaNsAreNotEqual);
    TEST_CHECK(testNaNIsNotAlmostEqualToNumber);
    TEST_CHECK(testIsNaN);
}

