/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
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

#pragma once

#include <stdint.h>

#include <string>

#include "nitf/TRE.hpp"
#include "nitf/exports.hpp"
#include "nitf/Property.hpp"

namespace nitf
{
    namespace TREs
    {
        template<typename T>
        struct NITRO_NITFCPP_API SetIndexedFieldProxy final
        {  
            nitf::TRE& tre_;
            std::string tag_;

            SetIndexedFieldProxy& operator=(const T& v)
            {
                tre_.setField(tag_, v);
                return *this;
            }
        };

        template<typename T>
        struct NITRO_NITFCPP_API IndexedField final
        {
            nitf::TRE& tre_;
            std::string name_;

            SetIndexedFieldProxy<T> operator[](size_t i)
            {
                auto tag = name_ + "[" + std::to_string(i) + "]";
                return SetIndexedFieldProxy<T>{ tre_, std::move(tag) };
            }
        };
    }
}


namespace nitf
{
    namespace TREs
    {
        class NITRO_NITFCPP_API ENGRDA final
        {
            nitf::TRE tre_;

            std::string get_A(const std::string& tag) const;
            int64_t get_N(const std::string& tag) const;

            void set_RECNT(int64_t, bool forceUpdate = true); // call updateFields()

        public:
            ENGRDA() noexcept(false);
            ~ENGRDA();
            ENGRDA(const ENGRDA&) = delete;
            ENGRDA& operator=(const ENGRDA&) = delete;
            ENGRDA(ENGRDA&&) = default;
            ENGRDA& operator=(ENGRDA&&) = default;

            // From ENGRDA.c
            //
            //static nitf_TREDescription description[] = {
            //    {NITF_BCS_A, 20, "Unique Source System Name", "RESRC" },
            Property<std::string> RESRC{ [&]() -> std::string { return get_A("RESRC"); }, [&](const std::string& v) -> void {  setField("RESRC", v); } };

            //    {NITF_BCS_N, 3, "Record Entry Count", "RECNT" },
            Property<int64_t> RECNT{ [&]() -> int64_t { return get_N("RECNT"); }, [&](int64_t v) -> void {  set_RECNT(v); } };

            //    {NITF_LOOP, 0, NULL, "RECNT"},
            //        {NITF_BCS_N, 2, "Engineering Data Label Length", "ENGLN" },
            //        /* This one we don't know the length of, so we have to use the special length tag */
            //        {NITF_BCS_A, NITF_TRE_CONDITIONAL_LENGTH, "Engineering Data Label",
            //                "ENGLBL", "ENGLN" },
            //        {NITF_BCS_N, 4, "Engineering Matrix Data Column Count", "ENGMTXC" },
            //        {NITF_BCS_N, 4, "Engineering Matrix Data Row Count", "ENGMTXR" },
            //        {NITF_BCS_A, 1, "Value Type of Engineering Data Element", "ENGTYP" },
            //        {NITF_BCS_N, 1, "Engineering Data Element Size", "ENGDTS" },
            IndexedField<int64_t> ENGDTS;

            //        {NITF_BCS_A, 2, "Engineering Data Units", "ENGDATU" },
            //        {NITF_BCS_N, 8, "Engineering Data Count", "ENGDATC" },
            IndexedField<int64_t> ENGDATC;

            //        /* This one we don't know the length of, so we have to use the special length tag */
            //        /* Notice that we use postfix notation to compute the length
            //         * We also don't know the type of data (it depends on ENGDTS), so
            //         * we need to override the TREHandler's read method.  If we don't do
            //         * this, not only will the field type potentially be wrong, but
            //         * strings will be endian swapped if they're of length 2 or 4. */
            //        {NITF_BINARY, NITF_TRE_CONDITIONAL_LENGTH, "Engineering Data",
            //                "ENGDATA", "ENGDATC ENGDTS *"},
            IndexedField<std::string> ENGDATA;

            //    {NITF_ENDLOOP, 0, NULL, NULL},
            //    {NITF_END, 0, NULL, NULL}
            //};


            void setField(const std::string& tag, const std::string& data, bool forceUpdate = false);
            void getField(const std::string& tag, std::string& data) const;
            void setField(const std::string& tag, int64_t, bool forceUpdate = false);
            void getField(const std::string& tag, int64_t&) const;

            void updateFields();
        };
    }
}
