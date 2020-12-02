#pragma once

#if __cplusplus >= 202002L // no C++20, see https://en.cppreference.com/w/cpp/preprocessor/replace
#include <span>
#else

#include <cstdint>

namespace std
{
	constexpr const std::size_t dynamic_extent = static_cast<std::size_t>(-1);
}

// GSL has gsl::span
#include "gsl/gsl.h"

namespace std
{
	// std::span<> is part of C++20.  Use our own implementation (rather, GSL's) until then.
	template<typename T, std::size_t Extent = dynamic_extent>
	using span = gsl::span<T, Extent>;
}

#endif