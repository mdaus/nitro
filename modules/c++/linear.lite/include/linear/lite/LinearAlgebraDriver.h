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

#ifndef __LINEAR_ALGEBRA_DRIVER_H__
#define __LINEAR_ALGEBRA_DRIVER_H__


/*!
 *  \file
 *  \brief Linear Algebra Driver for support module
 *
 *  Drivers can be defined using configure.  The
 *  defines addressed in this file occur automatically
 *  when configured to enable one package or another.
 *
 *  This header is very simple and simply typedefs
 *  creator patterns from the selected library, for
 *  the default implementations to use.
 *
 *  There are two ways to create a plugin, then.  One
 *  is to include this file and refer to the typedefs.
 *  However, it is also possible to have a separate
 *  linear algebra package within the plugin, as long as
 *  it links and includes those on its own, and uses the
 *  creator pattern from those drivers.
 *
 */
#if defined(HAVE_UBLAS_DRIVER)
#   include "linear/lite/ublas/MatrixUBlas.h"
#   include "linear/lite/ublas/VectorUBlas.h"
#define LINEAR_LITE_DRIVER "uBlas"
namespace linear
{
    namespace lite
    {
        typedef linear::lite::ublas::MatrixCreatorUBlas<double> MatrixCreator;
        typedef linear::lite::ublas::VectorCreatorUBlas<double> VectorCreator;
    }
}

#elif defined(HAVE_MTL_DRIVER)
#   include "linear/lite/mtl/MatrixMTL.h"
#   include "linear/lite/mtl/VectorMTL.h"
#define LINEAR_LITE_DRIVER "MTL"
namespace linear
{
    namespace lite
    {
        typedef linear::lite::mtl::MatrixCreatorMTL<double> MatrixCreator;
        typedef linear::lite::mtl::VectorCreatorMTL<double> VectorCreator;
    }
}
#elif defined(HAVE_TNT_DRIVER)
#   include "linear/lite/tnt/MatrixTNT.h"
#   include "linear/lite/tnt/VectorTNT.h"
#define LINEAR_LITE_DRIVER "TNT"
namespace linear
{
    namespace lite
    {
        typedef linear::lite::tnt::MatrixCreatorTNT<double> MatrixCreator;
        typedef linear::lite::tnt::VectorCreatorTNT<double> VectorCreator;
    }
}
#else
#    error "Require a linear algebra package"
#endif

#endif
