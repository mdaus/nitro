#include "pch.h"

#include "nrt_Test.h"

#define TEST_CASE(X) TEST(TestCaseName, X)
#define TEST_ASSERT(X) EXPECT_TRUE(X)
#define TEST_ASSERT_NULL(X) TEST_ASSERT((X) == NULL)

#define TEST_ASSERT_EQ(X1, X2) EXPECT_EQ(X1, X2)
#define TEST_ASSERT_EQ_INT(X1, X2) TEST_ASSERT_EQ(X1, X2)

#define TEST_MAIN(X)

#include "../nrt/unittests/test_list.c"
