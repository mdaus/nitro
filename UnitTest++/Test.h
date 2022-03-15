#pragma once

#include <stdint.h>
#include <string>

#include "CppUnitTest.h"

namespace Microsoft{ namespace VisualStudio {namespace CppUnitTestFramework
{
inline std::wstring ToString(const uint8_t& q)
{
	return std::to_wstring(q);
}
inline std::wstring ToString(const uint16_t& q)
{
	return std::to_wstring(q);
}
}}}

#define TEST_ASSERT(X) Assert::IsTrue(X)

template<typename T, typename U>
inline void test_assert_eq_(T&& t, U&& u)
{
	Assert::AreEqual(t, u);
}
#define TEST_ASSERT_EQ(X1, X2) { test_assert_eq_(X1, X2); test_assert_eq_(X2, X1); }
#define TEST_ASSERT_EQ_INT(X1, X2) TEST_ASSERT_EQ(X2, X1)
#define TEST_ASSERT_EQ_STR(X1, X2) TEST_ASSERT_EQ(X1, X2)
#define TEST_ASSERT_EQ_FLOAT(X1, X2) TEST_ASSERT_EQ(static_cast<float>(X1), static_cast<float>(X2))

#define TEST_ASSERT_NULL(X) Assert::IsNull((X))
#define TEST_ASSERT_TRUE(X) Assert::IsTrue((X))
#define TEST_ASSERT_FALSE(X) Assert::IsFalse((X))

template<typename T, typename U>
inline void test_assert_not_eq_(const T& t, const U& u)
{
	Assert::AreNotEqual(t, u);
}
#define TEST_ASSERT_NOT_EQ(X1, X2) { test_assert_not_eq_(X1, X2); test_assert_not_eq_(X2, X1);}

#define TEST_ASSERT_GREATER(X1, X2) EXPECT_GT(X1, X2)

#define TEST_ASSERT_ALMOST_EQ_EPS(X1, X2, EPS) Assert:AreEqual(X1, X2, EPS)
#define TEST_ASSERT_ALMOST_EQ(X1, X2) TEST_ASSERT_ALMOST_EQ_EPS(X1, X2, 0.0001)

#define TEST_ASSERT_EQ_MSG(msg, X1, X2) SCOPED_TRACE(msg); TEST_ASSERT_EQ(X1, X2)

#define TEST_EXCEPTION(X) EXPECT_ANY_THROW((X))
#define TEST_THROWS(X) EXPECT_ANY_THROW((X))

#define TEST_MAIN(X)
