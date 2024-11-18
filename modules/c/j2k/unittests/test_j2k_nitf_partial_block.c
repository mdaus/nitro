/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
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

#include <import/j2k.h>
#include <import/nitf.h>
#include <import/nrt.h>
#include <inttypes.h>

#include "Test.h"

const char multiband_j2k_nitf[] = {
        0x4e, 0x49, 0x54, 0x46, 0x30, 0x32, 0x2e, 0x31, 0x30, 0x30, 0x36, 0x42,
        0x46, 0x30, 0x31, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x30, 0x30,
        0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x7e, 0x7e, 0x7e,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x30, 0x30, 0x30, 0x32, 0x32, 0x39,
        0x38, 0x31, 0x33, 0x32, 0x35, 0x36, 0x30, 0x30, 0x30, 0x38, 0x36, 0x31,
        0x30, 0x30, 0x31, 0x30, 0x30, 0x30, 0x35, 0x32, 0x39, 0x30, 0x30, 0x30,
        0x30, 0x30, 0x30, 0x31, 0x39, 0x30, 0x35, 0x30, 0x30, 0x30, 0x30, 0x30,
        0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
        0x30, 0x30, 0x30, 0x30, 0x30, 0x34, 0x35, 0x37, 0x30, 0x30, 0x30, 0x47,
        0x45, 0x4f, 0x50, 0x53, 0x42, 0x30, 0x30, 0x34, 0x34, 0x33, 0x47, 0x45,
        0x4f, 0x44, 0x45, 0x47, 0x57, 0x6f, 0x72, 0x6c, 0x64, 0x20, 0x47, 0x65,
        0x6f, 0x64, 0x65, 0x74, 0x69, 0x63, 0x20, 0x53, 0x79, 0x73, 0x74, 0x65,
        0x6d, 0x20, 0x31, 0x39, 0x38, 0x34, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x57, 0x47, 0x45, 0x20, 0x57, 0x6f, 0x72, 0x6c, 0x64, 0x20, 0x47, 0x65,
        0x6f, 0x64, 0x65, 0x74, 0x69, 0x63, 0x20, 0x53, 0x79, 0x73, 0x74, 0x65,
        0x6d, 0x20, 0x31, 0x39, 0x38, 0x34, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x57, 0x45, 0x20, 0x47, 0x65, 0x6f, 0x64, 0x65, 0x74, 0x69, 0x63, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x47,
        0x45, 0x4f, 0x44, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
        0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x30, 0x30, 0x30, 0x30, 0x49, 0x4d, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x30,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x30, 0x30, 0x30, 0x30, 0x31, 0x30,
        0x36, 0x32, 0x30, 0x30, 0x30, 0x30, 0x31, 0x31, 0x31, 0x36, 0x49, 0x4e,
        0x54, 0x52, 0x47, 0x42, 0x20, 0x20, 0x20, 0x20, 0x20, 0x4d, 0x53, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x30, 0x38, 0x52, 0x47, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x30, 0x43, 0x38,
        0x56, 0x30, 0x33, 0x39, 0x33, 0x52, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x4e, 0x20, 0x20, 0x20, 0x30, 0x47, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x4e, 0x20, 0x20, 0x20, 0x30, 0x42, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x4e, 0x20, 0x20, 0x20, 0x30, 0x30, 0x42, 0x30, 0x30,
        0x30, 0x32, 0x30, 0x30, 0x30, 0x32, 0x31, 0x30, 0x32, 0x34, 0x31, 0x30,
        0x32, 0x34, 0x30, 0x38, 0x30, 0x30, 0x31, 0x30, 0x30, 0x30, 0x30, 0x30,
        0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x31, 0x2e, 0x30, 0x20,
        0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0xff, 0x4f,
        0xff, 0x51, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x00, 0x04, 0x5c, 0x00, 0x00,
        0x04, 0x26, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x03, 0x07, 0x01, 0x01, 0x07, 0x01, 0x01, 0x07, 0x01,
        0x01, 0xff, 0x52, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x13, 0x01, 0x05, 0x04,
        0x04, 0x00, 0x00, 0xff, 0x5c, 0x00, 0x23, 0x22, 0x77, 0x1e, 0x76, 0xea,
        0x76, 0xea, 0x76, 0xbc, 0x6f, 0x00, 0x6f, 0x00, 0x6e, 0xe2, 0x67, 0x4c,
        0x67, 0x4c, 0x67, 0x64, 0x50, 0x03, 0x50, 0x03, 0x50, 0x45, 0x57, 0xd2,
        0x57, 0xd2, 0x57, 0x61, 0xff, 0x90, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00,
        0x01, 0xf1, 0x00, 0x01, 0xff, 0x93, 0xc3, 0xe0, 0x20, 0x14, 0x00, 0x5c,
        0xaf, 0xff, 0x6f, 0x8f, 0x87, 0xc3, 0xe0, 0x20, 0x11, 0x50, 0x54, 0xaf,
        0xff, 0x7f, 0xf7, 0x9a, 0xcf, 0x98, 0x40, 0x11, 0x50, 0x54, 0xaf, 0xff,
        0x6f, 0x84, 0xca, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0xfc, 0x62, 0xc0, 0xe8, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x06, 0x12, 0x12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0xc2, 0xc0, 0x45, 0x28,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x09, 0x09, 0x7f, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0xfc, 0xc5, 0x00, 0x64, 0xe0, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x84, 0x84, 0xbf, 0xf5, 0x2c, 0xa0, 0x00, 0x00, 0x00, 0x00, 0x30,
        0x90, 0x97, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0xc5, 0x80, 0xff,
        0x77, 0xd2, 0x20, 0x00, 0x00, 0x00, 0x00, 0x01, 0x84, 0x84, 0xbf, 0xf5,
        0x2c, 0xa0, 0x00, 0x00, 0x00, 0x00, 0x30, 0x90, 0x97, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0xfc, 0x0b, 0x00, 0xfa, 0x93, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
        0x09, 0x09, 0x7f, 0xea, 0x50, 0x20, 0x00, 0x00, 0x00, 0x00, 0x61, 0x21,
        0x2f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0x63, 0x40, 0xff, 0x62,
        0x10, 0xa0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x90, 0x97, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xff, 0x90, 0x00,
        0x0a, 0x00, 0x01, 0x00, 0x00, 0x01, 0xc8, 0x00, 0x01, 0xff, 0x93, 0xc3,
        0xe0, 0x14, 0x14, 0x00, 0x5c, 0xab, 0x70, 0xc3, 0xe0, 0x14, 0x11, 0x50,
        0x54, 0xa6, 0x11, 0xcf, 0x98, 0x28, 0x11, 0x50, 0x54, 0xab, 0x8a, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0xfc, 0x62, 0x40, 0xcc, 0x1f, 0x00, 0x00, 0x00,
        0x00, 0x30, 0x90, 0x97, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0xc2, 0x00, 0xfd, 0x8a,
        0x00, 0x00, 0x00, 0x0c, 0x24, 0x25, 0xfc, 0xc3, 0x40, 0x1e, 0x24, 0x00,
        0x00, 0xc2, 0x42, 0x59, 0xbb, 0x90, 0x00, 0xc2, 0x42, 0x5f, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0xfc, 0xc3, 0x40, 0xd6, 0x7c, 0x40, 0x00, 0x0c,
        0x24, 0x25, 0x9b, 0xb9, 0x00, 0x0c, 0x24, 0x25, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc,
        0x07, 0x80, 0xfc, 0x71, 0x30, 0x00, 0x00, 0x01, 0x84, 0x84, 0xb2, 0xed,
        0x10, 0x01, 0x84, 0x84, 0xbe, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0x62, 0x40, 0xe2, 0xaf, 0x60, 0x00,
        0x00, 0x00, 0xc2, 0x42, 0x5f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xff, 0x90, 0x00,
        0x0a, 0x00, 0x02, 0x00, 0x00, 0x01, 0xbf, 0x00, 0x01, 0xff, 0x93, 0xc3,
        0xe0, 0x10, 0x01, 0xd9, 0x05, 0x7f, 0xc3, 0xe0, 0x10, 0x08, 0x91, 0x19,
        0x13, 0xcf, 0x98, 0x28, 0x08, 0x91, 0x19, 0x7c, 0x42, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0xfc, 0x62, 0x00, 0xfb, 0xc0, 0x00, 0x00, 0x00, 0x06, 0x12,
        0x12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0xfc, 0xc2, 0x40, 0xe3, 0xe9, 0xa0, 0x00, 0x00,
        0x00, 0xc2, 0x42, 0x5f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0xc3,
        0x00, 0xf3, 0x76, 0xc0, 0x00, 0x0c, 0x24, 0x23, 0x54, 0x28, 0x61, 0x21,
        0x2f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0xc2, 0xc0, 0xee, 0xbd,
        0x60, 0x01, 0x84, 0x84, 0x73, 0x05, 0x8c, 0x24, 0x25, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0xfc, 0x06, 0x00, 0x59, 0xf1, 0x00, 0x00, 0x30, 0x90, 0x8d, 0x50, 0xa1,
        0x84, 0x84, 0xbc, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0x62, 0x00,
        0xdd, 0x6d, 0x80, 0x00, 0x00, 0xc2, 0x42, 0x5d, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0xff, 0x90, 0x00, 0x0a, 0x00, 0x03, 0x00, 0x00, 0x01, 0x91, 0x00, 0x01,
        0xff, 0x93, 0xc3, 0xe0, 0x08, 0x01, 0xd5, 0xc3, 0x82, 0x08, 0x90, 0xc8,
        0x80, 0x08, 0x8e, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0xa0, 0x40, 0x7b,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0xf0, 0x20, 0x9c, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0x01, 0x00, 0xa6, 0x00,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xfc, 0xc0, 0xc0,
        0xd3, 0x11, 0x20, 0xfc, 0xc0, 0xc0, 0x40, 0x93, 0xa6, 0xf0, 0x40, 0x0c,
        0x6f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0xfc, 0xc0, 0x80, 0xa9, 0x1a, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0xfc, 0x01, 0x80, 0xf3, 0x5f, 0x4f, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0xfc, 0x60, 0x80, 0x29, 0xbe, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0xff, 0xd9};

TEST_CASE(test_multiband_j2k_nitf_partial_block)
{
    // Test reading a j2k-compressed, multiband NITF that contains partial
    // blocks

    nrt_Error error;
    nrt_IOInterface* io = NULL;
    nitf_Reader* reader = NULL;
    nitf_Record* record = NULL;
    uint8_t* pixbuf = NULL;

    printf("Buffer length: %ld\n", sizeof(multiband_j2k_nitf));
    io = nrt_BufferAdapter_construct(multiband_j2k_nitf,
                                     sizeof(multiband_j2k_nitf),
                                     0,
                                     &error);
    TEST_ASSERT(io);

    reader = nitf_Reader_construct(&error);
    TEST_ASSERT(reader);

    record = nitf_Reader_readIO(reader, io, &error);
    TEST_ASSERT(record);

    TEST_ASSERT_EQ_INT(nitf_Record_getNumImages(record, &error), 1);

    nitf_ListIterator iter = nitf_List_begin(record->images);
    nitf_ImageSegment* segment =
            (nitf_ImageSegment*)nitf_ListIterator_get(&iter);
    nitf_ImageSubheader* subheader = segment->subheader;

    TEST_ASSERT_EQ_STR(subheader->imageCompression->raw, "C8");

    j2k_Reader* j2kReader = NULL;
    j2k_Container* container = NULL;
    uint32_t cmpIt, nComponents;
    int i = 0;
    printf("Image %d contains J2K compressed data\n", (i + 1));
    printf("Offset: %" PRIu64 "\n", segment->imageOffset);
    TEST_ASSERT(nrt_IOInterface_seek(
            io, (nrt_Off)segment->imageOffset, NRT_SEEK_SET, &error));

    j2kReader = j2k_Reader_openIO(io, &error);
    TEST_ASSERT(j2kReader);

    container = j2k_Reader_getContainer(j2kReader, &error);
    TEST_ASSERT(container);

    unsigned int tileWidth = j2k_Container_getTileWidth(container, &error);
    unsigned int tileHeight = j2k_Container_getTileHeight(container, &error);
    TEST_ASSERT_EQ_INT(tileWidth, 1024);
    TEST_ASSERT_EQ_INT(tileHeight, 1024);

    unsigned int gridWidth = j2k_Container_getGridWidth(container, &error);
    unsigned int gridHeight = j2k_Container_getGridHeight(container, &error);
    TEST_ASSERT_EQ_INT(gridWidth, 1116);
    TEST_ASSERT_EQ_INT(gridHeight, 1062);
    TEST_ASSERT_EQ_INT(j2k_Container_getTilesX(container, &error), 2);
    TEST_ASSERT_EQ_INT(j2k_Container_getTilesY(container, &error), 2);
    printf("image type:\t%d\n", j2k_Container_getImageType(container, &error));

    nComponents = j2k_Container_getNumComponents(container, &error);
    printf("components:\t%d\n", nComponents);
    TEST_ASSERT_EQ_INT(nComponents, 3);

    for (cmpIt = 0; cmpIt < nComponents; ++cmpIt)
    {
        j2k_Component* c = j2k_Container_getComponent(container, cmpIt, &error);
        printf("===component %d===\n", (cmpIt + 1));
        printf("width:\t\t%d\n", j2k_Component_getWidth(c, &error));
        printf("height:\t\t%d\n", j2k_Component_getHeight(c, &error));
        printf("precision:\t%d\n", j2k_Component_getPrecision(c, &error));
        printf("x0:\t\t%d\n", j2k_Component_getOffsetX(c, &error));
        printf("y0:\t\t%d\n", j2k_Component_getOffsetY(c, &error));
        printf("x separation:\t%d\n", j2k_Component_getSeparationX(c, &error));
        printf("y separation:\t%d\n", j2k_Component_getSeparationY(c, &error));
    }

    uint32_t width, height;
    uint64_t bufSize;
    width = j2k_Container_getWidth(container, &error);
    height = j2k_Container_getHeight(container, &error);
    printf("width: %d\nheight: %d\n", width, height);

    // Sample NITF is broken into 4 blocks:
    //
    // +---------------------+-----+
    // |                     |     |
    // |                     |     |
    // |                     |     |
    // |                     |     |
    // |          A          |  B  |
    // |                     |     |
    // |                     |     |
    // |                     |     |
    // +---------------------+-----+
    // |                     |     |
    // |          C          |  D  |
    // |                     |     |
    // +---------------------+-----+
    //
    // Where:
    // - block A is a full block (1024r x 1024c),
    // - block B is narrower than a full block (1024r x 92c)
    // - block C is shorter than a full block (38r x 1024c)
    // - block D is narrower and shorter (38r x 92c)

    // ************ Block A ************
    int npix = 1024 * 1024;
    bufSize =
            j2k_Reader_readRegion(j2kReader, 0, 0, 1024, 1024, &pixbuf, &error);
    TEST_ASSERT(bufSize >= 3 * npix);
    TEST_ASSERT(pixbuf != NULL);

    uint8_t* pixptr = pixbuf;
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 20);
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 217);
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 101);
    free(pixbuf);
    pixbuf = NULL;

    // ************ Block B ************
    npix = 1024 * 92;
    bufSize = j2k_Reader_readRegion(
            j2kReader, 1024, 0, 1116, 1024, &pixbuf, &error);
    TEST_ASSERT(bufSize >= 3 * npix);
    TEST_ASSERT(pixbuf != NULL);

    pixptr = pixbuf;
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 20);
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 217);
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 101);
    free(pixbuf);
    pixbuf = NULL;

    // ************ Block C ************
    npix = 38 * 1024;
    bufSize = j2k_Reader_readRegion(
            j2kReader, 0, 1024, 1024, 1062, &pixbuf, &error);
    TEST_ASSERT(bufSize >= 3 * npix);
    TEST_ASSERT(pixbuf != NULL);

    pixptr = pixbuf;
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 20);
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 217);
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 101);
    free(pixbuf);
    pixbuf = NULL;

    // ************ Block D ************
    npix = 38 * 92;
    bufSize = j2k_Reader_readRegion(
            j2kReader, 1024, 1024, 1116, 1062, &pixbuf, &error);
    TEST_ASSERT(bufSize >= 3 * npix);
    TEST_ASSERT(pixbuf != NULL);

    pixptr = pixbuf;
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 20);
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 217);
    for (int i = 0; i < npix; i++)
        TEST_ASSERT_EQ_INT(*pixptr++, 101);
    free(pixbuf);
    pixbuf = NULL;

    j2k_Reader_destruct(&j2kReader);
    nitf_Record_destruct(&record);
    nitf_Reader_destruct(&reader);
    nrt_IOInterface_destruct(&io);
}

TEST_MAIN(CHECK(test_multiband_j2k_nitf_partial_block);)