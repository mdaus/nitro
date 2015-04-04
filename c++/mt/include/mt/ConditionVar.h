/* =========================================================================
 * This file is part of mt-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * mt-c++ is free software; you can redistribute it and/or modify
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


#ifndef __MT_CONDITION_VAR_H__
#define __MT_CONDITION_VAR_H__

#    if defined(USE_NSPR_THREADS)
#        include "mt/ConditionVarNSPR.h"
namespace mt
{
typedef ConditionVarNSPR ConditionVar;
}
#    elif defined(WIN32)
#        include "mt/ConditionVarWin32.h"
namespace mt
{
typedef ConditionVarWin32 ConditionVar;
}
#    elif defined(__sun)
#        include "mt/ConditionVarSolaris.h"
namespace mt
{
typedef ConditionVarSolaris ConditionVar;
}
#    elif defined(__sgi)
#        include "mt/ConditionVarIrix.h"
namespace mt
{
typedef ConditionVarIrix ConditionVar;
}
//default to POSIX
#    else
#        include "mt/ConditionVarPosix.h"
namespace mt
{
typedef ConditionVarPosix ConditionVar;
}
#    endif // Which thread package?

#endif // End of header

