/* =========================================================================
 * This file is part of sys-c++
 * =========================================================================
 *
 * (C) Copyright 2020, Maxar Technologies, Inc.
 *
 * sys-c++ is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; If not, http://www.gnu.org/licenses/.
 *
 */
#ifndef CODA_OSS_sys_Filesystem_h_INCLUDED_
#define CODA_OSS_sys_Filesystem_h_INCLUDED_
#pragma once

//
// For now, these are just some wrappers around the CODA-OSS routines ...
// eventually, this should be replaced with C++17's <filesystem>.
//
// This does not even TRY to be a complete implementation of std::filesystem.
//

#include <string>
#include <ostream>

#include "Conf.h"

namespace sys
{
// http://en.cppreference.com/w/cpp/filesystem
namespace Filesystem
{
  struct path; // forward
  namespace details
  {
    extern bool Equals(const path& lhs, const path& rhs) noexcept;
    extern std::ostream& Ostream(std::ostream& os, const path& p);
  }

// http://en.cppreference.com/w/cpp/filesystem/path
struct path final // N.B. this is an INCOMPLETE and NON-STANDARD implementation!
{
    // character type used by the native encoding of the filesystem: char on POSIX, wchar_t on Windows
    #ifdef _WIN32
    using value_type = wchar_t;
    #else
    using value_type = char;
    #endif
    using string_type = std::basic_string<value_type>;

    private:

    public:
    // http://en.cppreference.com/w/cpp/filesystem/path/path
    path() noexcept;
    path(const path&);
    path(const string_type&);
    template<typename TSource>
    path(const TSource& source)
    {
        p_ = to_native(source);
    }

    path& operator/=(const path&);  // http://en.cppreference.com/w/cpp/filesystem/path/append
    template <typename TSource>
    path& operator/=(const TSource& source)  // http://en.cppreference.com/w/cpp/filesystem/path/append
    {
        return (*this) /= path(to_native(source));
    }
    void clear() noexcept;  // http://en.cppreference.com/w/cpp/filesystem/path/clear

    // http://en.cppreference.com/w/cpp/filesystem/path/native
    const value_type* c_str() const noexcept;
    const string_type& native() const noexcept;
    operator string_type() const;

    std::string string() const;  // http://en.cppreference.com/w/cpp/filesystem/path/string

    path parent_path() const;  // http://en.cppreference.com/w/cpp/filesystem/path/parent_path
    path filename() const;  // http://en.cppreference.com/w/cpp/filesystem/path/filename
    path stem() const;  // http://en.cppreference.com/w/cpp/filesystem/path/stem
    path extension() const;  // https://en.cppreference.com/w/cpp/filesystem/path/extension

    bool empty() const noexcept;  // http://en.cppreference.com/w/cpp/filesystem/path/empty
    bool is_absolute() const;  // http://en.cppreference.com/w/cpp/filesystem/path/is_absrel
    bool is_relative() const;  // http://en.cppreference.com/w/cpp/filesystem/path/is_absrel

    friend bool operator==(const path& lhs, const path& rhs) noexcept  // https://en.cppreference.com/w/cpp/filesystem/path/operator_cmp
    {
      return details::Equals(lhs, rhs);
    }
    friend bool operator!=(const path& lhs, const path& rhs) noexcept  // https://en.cppreference.com/w/cpp/filesystem/path/operator_cmp
    {
      return !(lhs == rhs);
    }
    friend std::ostream& operator<<(std::ostream& os, const path& p) // https://en.cppreference.com/w/cpp/filesystem/path/operator_ltltgtgt
    {
      return details::Ostream(os, p);
    }

private:
    string_type p_;
    static string_type to_native(const std::string& s);
};

path operator/(const path& lhs, const path& rhs);  // http://en.cppreference.com/w/cpp/filesystem/path/operator_slash

path absolute(const path&);  // http://en.cppreference.com/w/cpp/filesystem/absolute
bool create_directory(const path&);  // https://en.cppreference.com/w/cpp/filesystem/create_directory
path current_path();  // https://en.cppreference.com/w/cpp/filesystem/current_path
bool remove(const path& p);  // https://en.cppreference.com/w/cpp/filesystem/remove
path temp_directory_path();  // https://en.cppreference.com/w/cpp/filesystem/temp_directory_path

bool is_regular_file(const path& p);  // https://en.cppreference.com/w/cpp/filesystem/is_regular_file
bool is_directory(const path& p);  // https://en.cppreference.com/w/cpp/filesystem/is_directory
bool exists(const path& p);  // https://en.cppreference.com/w/cpp/filesystem/exists
}
}

#ifndef CODA_OSS_DEFINE_std_filesystem_
    #if CODA_OSS_cpp17
        // Some versions of G++ say they're C++17 but don't have <filesystem>
        #if __has_include(<filesystem>) && (__cpp_lib_filesystem >= 201703)
            #define CODA_OSS_DEFINE_std_filesystem_ -1  // OK to #include <>, below
        #else
            #define CODA_OSS_DEFINE_std_filesystem_ 1 // must have std::filesystem w/C++17
        #endif // __has_include
    #else
        #define CODA_OSS_DEFINE_std_filesystem_ CODA_OSS_AUGMENT_std_namespace  // maybe use our own
    #endif
#endif  // CODA_OSS_DEFINE_std_filesystem_


#if CODA_OSS_DEFINE_std_filesystem_ == 1
    namespace std // This is slightly uncouth: we're not supposed to augment "std".
    {
        namespace filesystem = ::sys::Filesystem;
    }
    #define CODA_OSS_lib_filesystem 1
#elif CODA_OSS_DEFINE_std_filesystem_ == -1 // set above
    #include <filesystem>
    #define CODA_OSS_lib_filesystem 1
#endif  // CODA_OSS_DEFINE_std_filesystem_

#if CODA_OSS_cpp17 || ( CODA_OSS_DEFINE_std_filesystem_ == 1) // sanity check on above
#if !CODA_OSS_lib_filesystem
#error must have std::filesystem with C++17
#endif // CODA_OSS_lib_filesystem
#endif

// coda_oss::filesystem will always work, and will be std::filesystem if available.
namespace coda_oss
{
    #if CODA_OSS_lib_filesystem
    namespace filesystem = ::std::filesystem;
    #else
    namespace filesystem = ::sys::Filesystem;
    #endif
}

#endif  // CODA_OSS_sys_Filesystem_h_INCLUDED_
