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

#include "TREField.hpp"

namespace nitf
{
    namespace TREs
    {
        class NITRO_NITFCPP_API ENGRDA final
        {
            nitf::TRE tre_;

        public:
            ENGRDA(const std::string& id = "") noexcept(false);
            ~ENGRDA();
            ENGRDA(const ENGRDA&) = delete;
            ENGRDA& operator=(const ENGRDA&) = delete;
            ENGRDA(ENGRDA&&) = default;
            ENGRDA& operator=(ENGRDA&&) = delete;

            // From ENGRDA.c
            //
            //static nitf_TREDescription description[] = {
            //    {NITF_BCS_A, 20, "Unique Source System Name", "RESRC" },
            TREField_BCS_A<20> RESRC;

            //    {NITF_BCS_N, 3, "Record Entry Count", "RECNT" },
            TREField_BCS_N<3> RECNT;

            //    {NITF_LOOP, 0, NULL, "RECNT"},
            //        {NITF_BCS_N, 2, "Engineering Data Label Length", "ENGLN" },
            //        /* This one we don't know the length of, so we have to use the special length tag */
            //        {NITF_BCS_A, NITF_TRE_CONDITIONAL_LENGTH, "Engineering Data Label",
            //                "ENGLBL", "ENGLN" },
            //        {NITF_BCS_N, 4, "Engineering Matrix Data Column Count", "ENGMTXC" },
            //        {NITF_BCS_N, 4, "Engineering Matrix Data Row Count", "ENGMTXR" },
            //        {NITF_BCS_A, 1, "Value Type of Engineering Data Element", "ENGTYP" },
            //        {NITF_BCS_N, 1, "Engineering Data Element Size", "ENGDTS" },
            IndexedField<TREField_BCS_N<1>> ENGDTS;

            //        {NITF_BCS_A, 2, "Engineering Data Units", "ENGDATU" },
            //        {NITF_BCS_N, 8, "Engineering Data Count", "ENGDATC" },
            IndexedField<TREField_BCS_N<8>> ENGDATC;

            //        /* This one we don't know the length of, so we have to use the special length tag */
            //        /* Notice that we use postfix notation to compute the length
            //         * We also don't know the type of data (it depends on ENGDTS), so
            //         * we need to override the TREHandler's read method.  If we don't do
            //         * this, not only will the field type potentially be wrong, but
            //         * strings will be endian swapped if they're of length 2 or 4. */
            //        {NITF_BINARY, NITF_TRE_CONDITIONAL_LENGTH, "Engineering Data",
            //                "ENGDATA", "ENGDATC ENGDTS *"},
            IndexedField<TREField_BCS_A<>> ENGDATA;

            //    {NITF_ENDLOOP, 0, NULL, NULL},
            //    {NITF_END, 0, NULL, NULL}
            //};

            template <typename T>
            void setFieldValue(const std::string& tag, const T& value, bool forceUpdate = false)
            {
                tre_.setFieldValue(tag, value, forceUpdate);
            }
            void setFieldValue(const std::string& tag, const void* data, size_t dataLength, bool forceUpdate = false)
            {
                tre_.setFieldValue(tag, data, dataLength, forceUpdate);
            }

            template<typename T>
            const T& getFieldValue(const std::string& tag, T& value) const
            {
                return tre_.getFieldValue(tag, value);
            }
            template<typename T>
            const T getFieldValue(const std::string& tag) const
            {
                return tre_.getFieldValue<T>(tag);
            }

            void updateFields();
        };
    }
}
