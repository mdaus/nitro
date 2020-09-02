#pragma once

#include <cstddef>

#if __cplusplus < 201703L // C++17

// https://en.cppreference.com/w/cpp/types/byte
namespace std
{
	enum class byte : unsigned char { };
}

#endif // __cplusplus