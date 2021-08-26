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

#include "nitf/TREs/ENGRDA.hpp"

using namespace nitf;

TREs::ENGRDA::ENGRDA()
	: tre_("ENGRDA", "ENGRDA") // what's the difference between "tag" and "id"?
{
}

TREs::ENGRDA::~ENGRDA() = default;

void TREs::ENGRDA::setField(const std::string& tag, const std::string& data, bool forceUpdate)
{
	tre_.setField(tag, data, forceUpdate);
}
void TREs::ENGRDA::setField(const std::string& tag, int64_t data, bool forceUpdate)
{
	tre_.setField(tag, data, forceUpdate);
}
std::string TREs::ENGRDA::get_A(const std::string& tag) const
{
	return tre_.getField(tag);

}
void TREs::ENGRDA::getField(const std::string& tag, std::string& data) const
{
	data = get_A(tag);
}

int64_t TREs::ENGRDA::get_N(const std::string& tag) const
{
	return tre_.getField(tag);
}
void TREs::ENGRDA::getField(const std::string& tag, int64_t& data) const
{
	data = get_N(tag);
}

void TREs::ENGRDA::set_RECNT(int64_t data, bool forceUpdate /*=true*/) 
{
	setField("RECNT", data, forceUpdate);
}


void TREs::ENGRDA::updateFields()
{
	tre_.updateFields();
}