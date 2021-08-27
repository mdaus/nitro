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
#include <assert.h>

#include <string>
#include <stdexcept>
#include <new> // std::nothrow

#include "nitf/TRE.hpp"
#include "nitf/exports.hpp"

namespace nitf
{
    namespace TREs
    {
        namespace details
        {
            // A simple wrapper around TRE::setFieldValue() and TRE::getFieldValue().
            // Everything to make the call is in one spot.  More importantly, it turns
            // and assignment into a call to setFieldValue() and a cast calls getFieldValue()
            template<typename T>
            struct const_field final
            {
                const TRE& tre_;
                std::string key_;

                const_field(const TRE& tre, const std::string& key) : tre_(tre), key_(key) {}
                const T getFieldValue() const
                {
                    return tre_.getFieldValue<T>(key_);
                }
                const T value() const
                {
                    return getFieldValue();
                }
                operator const T() const
                {
                    return value();
                }
            };
            template<typename T>
            struct field final
            {
                const_field<T> field_;
                TRE& tre_; // need non-const
                bool forceUpdate_;

                field(TRE& tre, const std::string& key, bool forceUpdate = false) : field_(tre, key), tre_(tre), forceUpdate_(forceUpdate) {}
                void setFieldValue(const T& v)
                {
                    tre_.setFieldValue(field_.key_, v, forceUpdate_);
                }
                void operator=(const T& v)
                {
                    setFieldValue(v);
                }

                const T getFieldValue() const
                {
                    return field_.getFieldValue();
                }
                const T value() const
                {
                    return getFieldValue();
                }
                operator const T() const
                {
                    return value();
                }
            };
        }


        // Include the size of the TRE field to make the types more unique.
        // Maybe check it against what's reported at run-time?
        template<nitf_FieldType, size_t sz, typename T>
        class TREField final
        {
            details::field<T> field_;
        public:
            using value_type = T;
            static constexpr size_t size = sz;

            TREField(TRE& tre, const std::string& key, bool forceUpdate = false) : field_(tre, key, forceUpdate) {}

            void operator=(const value_type& v)
            {
                field_ = v;
            }

            const value_type value() const
            {
                return field_.value();
            }
            operator const value_type() const
            {
                return value();
            }
        };

        template<typename size_t sz = SIZE_MAX>
        using TREField_BCS_A = TREField<NITF_BCS_A, sz, std::string>;

        template<typename size_t sz, typename T = int64_t>
        using TREField_BCS_N = TREField<NITF_BCS_N, sz, T>;

        template<typename TTREField_BCS>
        class IndexedField final
        {
            TRE& tre_;
            std::string name_;
            std::string indexFieldName_;

            std::string make_tag(size_t i, std::nothrow_t) const
            {
                return name_ + "[" + std::to_string(i) + "]";
            }
            std::string make_tag(size_t i) const
            {
                auto retval = make_tag(i, std::nothrow); // don't duplicate code in throw(), below
                if (i < size()) // better than calling exists() on a non-existent field?
                {
                    if (tre_.exists(retval))
                    {
                        return retval;
                    }
                }
                throw std::out_of_range("tag '" + retval + "' does not exist.");
            }

            TTREField_BCS make_field(size_t i, std::nothrow_t) const
            {
                return TTREField_BCS(tre_, make_tag(i, std::nothrow));
            }
            TTREField_BCS make_field(size_t i) const
            {
                return TTREField_BCS(tre_, make_tag(i));
            }

            using value_type = typename TTREField_BCS::value_type;
        public:
            // Indexed fields have their size set by another field.  Rather than trying to get (really)
            // fancy as pass a TREField<>, just use another name.
            IndexedField(TRE& tre, const std::string& name, const std::string& indexFieldName)
                : tre_(tre), name_(name), indexFieldName_(indexFieldName)
            {
                (void)size(); // be sure the index field exists
            }
            ~IndexedField() = default;
            IndexedField(const IndexedField&) = delete;
            IndexedField& operator=(const IndexedField&) = delete;
            IndexedField(IndexedField&&) = default;
            IndexedField& operator=(IndexedField&&) = delete;

            TTREField_BCS operator[](size_t i)
            {
                return make_field(i, std::nothrow);
            }
            TTREField_BCS at(size_t i) // c.f. std::vector
            {
                return make_field(i);
            }

            // Directly return the underlying value rather than a TTREField_BCS
            value_type operator[](size_t i) const
            {
                return make_field(i, std::nothrow).value();
            }
            value_type at(size_t i) const // c.f. std::vector
            {
                return make_field(i).value();
            }

            size_t size() const // c.f. std::vector
            {
                // don't save this value as it could change after updateFields()
                return details::const_field<size_t>(tre_, indexFieldName_);
            }
            bool empty() const // c.f. std::vector
            {
                return size() == 0;
            }
        };
    }
}

