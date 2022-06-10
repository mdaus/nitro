/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2021, Maxar Technologies, Inc.
 *
 * NITRO is free software; you can redistribute it and/or modify
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
 * License along with this program; if not, If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef NITF_Enum_hpp_INCLUDED_
#define NITF_Enum_hpp_INCLUDED_
#pragma once

#include <string>
#include <map>
#include <stdexcept>
#include <ostream>
#include <std/optional>

#include "str/Manip.h"
#include "str/EncodedStringView.h"

namespace nitf
{
    namespace details
    {
        template<typename TKey, typename TValue>
        inline std::map<TValue, TKey> swap_key_value(const std::map<TKey, TValue>& map)
        {
            std::map<TValue, TKey> retval;
            for (const auto& p : map)
            {
                retval[p.second] = p.first;
            }
            return retval;
        }

        template<typename TKey, typename TValue>
        inline std::optional<TValue> optional_index(const std::map<TKey, TValue>& map, const TKey& key)
        {
            const auto it = map.find(key);
            return it == map.end() ? std::optional<TValue>() : std::optional<TValue>(it->second);
        }
        template<typename TKey, typename TValue, typename TException>
        inline std::optional<TValue> index(const std::map<TKey, TValue>& map, const TKey& key, const TException* pEx) noexcept(false)
        {
            auto result = optional_index(map, key);
            if (!result.has_value() && (pEx != nullptr))
            {
                throw *pEx;
            }
            return result;
        }

        template<typename TKey, typename TValue, typename TException>
        inline TValue index(const std::map<TKey, TValue>& map, const TKey& key, const TException& ex) noexcept(false)
        {
            const auto result = index(map, key, &ex);
            return *result;
        }
        template<typename TKey, typename TValue>
        inline TValue index(const std::map<TKey, TValue>& map, const TKey& key) noexcept(false)
        {
            return index(map, key, std::invalid_argument("key not found in map."));
        }

        template<typename T, typename TException>
        inline std::optional<std::string> to_string(T v, const TException* pEx, const std::map<std::string, T>& string_to_enum) noexcept(false)
        {
            static const auto enum_to_string = details::swap_key_value(string_to_enum);
            return index(enum_to_string, v, pEx);
        }

        template<typename T, typename TException>
        inline std::optional<T> from_string(std::string v, const TException* pEx, const std::map<std::string, T>& string_to_enum) noexcept(false)
        {
            str::trim(v);
            return index(string_to_enum, v, pEx);
        }
    }

#define NITF_ENUM_define_enum_(name, ...) enum class name { __VA_ARGS__ }

#define NITF_ENUM_map_entry(name, n) { #n, name::n }
#define NITF_ENUM_map_entry_2(name, n1, n2) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry(name, n2)
#define NITF_ENUM_map_entry_3(name, n1, n2, n3) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_2(name, n2, n3)
#define NITF_ENUM_map_entry_4(name, n1, n2, n3, n4)  NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_3(name, n2, n3, n4)
#define NITF_ENUM_map_entry_5(name, n1, n2, n3, n4, n5)  NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_4(name, n2, n3, n4, n5)
#define NITF_ENUM_map_entry_6(name, n1, n2, n3, n4, n5, n6)  NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_5(name, n2, n3, n4, n5, n6)
#define NITF_ENUM_map_entry_7(name, n1, n2, n3, n4, n5, n6, n7) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_6(name, n2, n3, n4, n5, n6, n7)
#define NITF_ENUM_map_entry_8(name, n1, n2, n3, n4, n5, n6, n7, n8) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_7(name, n2, n3, n4, n5, n6, n7, n8)
#define NITF_ENUM_map_entry_9(name, n1, n2, n3, n4, n5, n6, n7, n8, n9) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_8(name, n2, n3, n4, n5, n6, n7, n8, n9)
#define NITF_ENUM_map_entry_10(name, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_9(name, n2, n3, n4, n5, n6, n7, n8, n9, n10)
#define NITF_ENUM_map_entry_11(name, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_10(name, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11)
#define NITF_ENUM_map_entry_12(name, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_11(name, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12)
#define NITF_ENUM_map_entry_13(name, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_12(name, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13)
#define NITF_ENUM_map_entry_14(name, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_13(name, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14)
#define NITF_ENUM_map_entry_15(name, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15) NITF_ENUM_map_entry(name, n1), NITF_ENUM_map_entry_14(name, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15)

#define NITF_ENUM_define_string_to_enum_ostream_(name) inline std::ostream& operator<<(std::ostream& os, name e) { os << to_string(e); return os; }
#define NITF_ENUM_define_string_to_enum_begin(name)  NITF_ENUM_define_string_to_enum_ostream_(name) \
   inline const std::map<std::string, name>& string_to_enum(name) { static const std::map<std::string, name> retval {
#define NITF_ENUM_define_string_to_end }; return retval; }
#define NITF_ENUM_define_string_to_enum_(name, ...) NITF_ENUM_define_string_to_enum_begin(name) __VA_ARGS__  \
    NITF_ENUM_define_string_to_end

#define NITF_ENUM(n, name, ...) NITF_ENUM_define_enum_(name, __VA_ARGS__); \
        NITF_ENUM_define_string_to_enum_(name, NITF_ENUM_map_entry_##n(name, __VA_ARGS__))


    template<typename T>
    inline std::string to_string(T v, const std::map<std::string, T>& string_to_enum)
    {
        static const auto enum_to_string = details::swap_key_value(string_to_enum);
        return details::index(enum_to_string, v);
    }
    template<typename T>
    inline std::string to_string(T v)
    {
        return to_string(v, string_to_enum(T()));
    }

    template<typename T>
    inline std::wstring to_wstring(T v, const std::map<std::string, T>& string_to_enum)
    {
        return str::EncodedStringView(to_string(v, string_to_enum)).wstring();
    }
    template<typename T>
    inline std::wstring to_wstring(T v)
    {
        return str::EncodedStringView(to_string(v)).wstring();
    }

    template<typename T>
    inline T from_string(std::string v, const std::map<std::string, T>& string_to_enum) noexcept(false)
    {
        str::trim(v);
        return details::index(string_to_enum, v);
    }
    template<typename T>
    inline T from_string(const std::string& v) noexcept(false)
    {
        return from_string<T>(v, string_to_enum(T()));
    }
}
#endif // NITF_Enum_hpp_INCLUDED_
