/* =========================================================================
 * This file is part of CODA-OSS
 * =========================================================================
 * 
 * C) Copyright 2004 - 2016, MDA Information Systems LLC
 * 
 * CODA-OSS is free software; you can redistribute it and/or modify
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

#ifndef __NPUTILS_NUMPYHELPER_H
#define __NPUTILS_NUMPYHELPER_H

#include <numpy/arrayobject.h>
#include <types/RowCol.h>
#include <except/Exception.h>

namespace nputils
{

/*! Verifies that the input object is a PyArray object */
void verifyArray(PyObject *pyObject)
{
    if (!PyArray_Check(pyObject))
    {
        throw except::Exception("Invalid data type (expected numpy array)");
    }
}

/* Verifies that the input object element type matches the input typeNum) */
void verifyType(PyObject* pyObject, int typeNum)
{
    if (PyArray_TYPE(reinterpret_cast<PyArrayObject*>(pyObject)) != typeNum)
    {
        throw except::Exception("Unexpected numpy array element type");
    }
}

/*! Verifies both that the input object is a PyArray and that its element type matches the input typeNume */
void verifyArrayType(PyObject *pyObject, int typeNum)
{
    verifyArray(pyObject);
    verifyType(pyObject, typeNum);
}

/*! Returns array dimensions and enforces a dimension check of two*/
const npy_intp* const getDimensions(PyObject* pyArrayObject)
{
    verifyArray(pyArrayObject);
    int ndims = PyArray_NDIM(reinterpret_cast<PyArrayObject*>(pyArrayObject));
    if (ndims != 2)
    {
        throw except::Exception("Numpy array has dimensions different than 2");
        return 0;
    }
    return PyArray_DIMS(reinterpret_cast<PyArrayObject*>(pyArrayObject));
}

/*! Variant returning types::RowCol<size_t> version of dimensions for convenience */
types::RowCol<size_t> getDimensionsRC(PyObject* pyArrayObject)
{
   const npy_intp* dims = getDimensions(pyArrayObject);
   return types::RowCol<size_t>(dims[0], dims[1]);
}


/*! Verifies that the objects are of the same dimensions */
void verifyObjectsAreOfSameDimensions(PyObject* pObject1, PyObject* pObject2)
{
    const npy_intp* const dimObj1 = getDimensions(pObject1);
    const npy_intp* const dimObj2 = getDimensions(pObject2);

    if(dimObj1[0] != dimObj2[0] || dimObj1[1] != dimObj2[1])
    {
        throw except::Exception("Numpy arrays are of differing dimensions");
    }
}


/*! Helper function used to either verify that an object is either an array with the requested dimensions and type
 * OR create a new array of the requested dimensions and type, if not. */
void createOrVerify(PyObject*& pyObject, int typeNum, size_t rows, size_t cols)
{
    if (pyObject == Py_None) // none passed in-- so create new
    {
        npy_intp odims[2] = {rows, cols};
        pyObject = PyArray_SimpleNew(2, odims, typeNum);
    }
    else
    {
        verifyArrayType(pyObject, typeNum);
        const npy_intp* const outdims = getDimensions(pyObject);
        if (outdims[0] != rows  || outdims[1] != cols)
        {
            throw except::Exception("Desired array does not match required row, cols");
        }
    }
}



/*! Verifies Array Type and TypeNum for input and output.  If output array is Py_None, constructs a new PyArray of the desired specifications
 *  If rows and cols are default (-1) the output PyArray will be constructed with the size of the input PyArray. */
void prepareInputAndOutputArray(PyObject* pyInObject, PyObject*& pyOutObject, int inputTypeNum, int outputTypeNum, int rows = -1, int cols = -1)
{
    verifyArrayType(pyInObject, inputTypeNum);

    const npy_intp* const dims = getDimensions(pyInObject);

    int desired_rows = (rows != -1) ? rows : dims[0];
    int desired_cols = (cols != -1) ? cols : dims[1];

    createOrVerify(pyOutObject, outputTypeNum, desired_rows, desired_cols);
}

/*! Extract PyArray Buffer as raw array of type T* */
template<typename T>
T* getBuffer(PyObject* pyObject)
{
    return reinterpret_cast<T*>(PyArray_BYTES(reinterpret_cast<PyArrayObject*>(pyObject)));
}

}
#endif
