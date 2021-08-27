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

// from TRE::getID()
/**
 * Get the TRE identifier. This is NOT the tag, however it may be the
 * same value as the tag. The ID is used to identify a specific
 * version/incarnation of the TRE, if multiple are possible. For most TREs,
 * this value will be the same as the tag.
 */
TREs::ENGRDA::ENGRDA(const std::string& id)
	: tre_("ENGRDA", id.empty() ? "ENGRDA" : id.c_str()),
	RESRC(tre_, "RESRC"),
	RECNT(tre_, "RECNT", true /*forceUpdate*/),
	ENGDTS{ tre_, "ENGDTS" },
	ENGDATC{ tre_, "ENGDATC" },
	ENGDATA{ tre_, "ENGDATA" }
{
}
TREs::ENGRDA::~ENGRDA() = default;

void TREs::ENGRDA::updateFields()
{
	tre_.updateFields();
}