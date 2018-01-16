/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2018, Ball Aerospace
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


#include <import/nitf.h>

NITF_CXX_GUARD

static nitf_TREDescription description[] = {
{NITF_BCS_N,   2,     "Number of relationship types.", "NUM_GROUPS"},
{NITF_LOOP,    0,     NULL,                            "NUM_GROUPS"},
{NITF_BCS_A,  32,     "Mate Relationship Type",        "RELATIONSHIP"},
{NITF_BCS_N,   3,     "Number of Mates",               "NUM_MATES"},
{NITF_LOOP,    0,     NULL,                            "NUM_MATES"},
{NITF_BCS_A,  42,     "Mate Source",                   "SOURCE"},
{NITF_BCS_A,  20,     "Identifier Type",               "ID_TYPE"},
{NITF_BCS_A, 256,     "Mate File Identifier",          "MATE_ID"},
{NITF_ENDLOOP, 0,     NULL,                            NULL}, /* NUM_MATES */
{NITF_ENDLOOP, 0,     NULL,                            NULL}, /* NUM_GROUPS */
{NITF_END,     0,     NULL,                            NULL}
};

NITF_DECLARE_SINGLE_PLUGIN(MATESA, description)

NITF_CXX_ENDGUARD

