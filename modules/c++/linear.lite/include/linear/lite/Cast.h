/* =========================================================================
 * This file is part of linear.lite-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * linear.lite-c++ is free software; you can redistribute it and/or modify
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
 * License along with this program; If not, 
 * see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef __LINALG_LITE_CAST_H__
#define __LINALG_LITE_CAST_H__

#ifdef LINALG_LITE_NO_RTTI_CHECK
#else
#    include <typeinfo>
#endif

namespace linear
{
	namespace lite
	{
#ifdef LINALG_LITE_NO_RTTI_CHECK
     template<typename T, typename R> T down_cast(R r) { return static_cast<T>(r); }
#else
     template<typename T, typename R> T down_cast(R r) { return dynamic_cast<T>(r); }
#endif
	}
}




#endif
