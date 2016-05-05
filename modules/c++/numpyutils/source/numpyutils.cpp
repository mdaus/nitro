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

#include <numpyutils/numpyutils.h>

#include <except/Exception.h>
#include <sys/Conf.h>

namespace numpyutils
{

void verifyArray(PyObject *pyObject)
{
    if (!PyArray_Check(pyObject))
    {
        throw except::Exception(Ctxt(
                    "Invalid data type (expected numpy array)"));
    }
}

void verifyType(PyObject* pyObject, int typeNum)
{
    if (PyArray_TYPE(reinterpret_cast<PyArrayObject*>(pyObject)) != typeNum)
    {
        throw except::Exception(Ctxt("Unexpected numpy array element type"));
    }
}

void verifyArrayType(PyObject *pyObject, int typeNum)
{
    verifyArray(pyObject);
    verifyType(pyObject, typeNum);
}

const npy_intp* const getDimensions(PyObject* pyArrayObject)
{
    verifyArray(pyArrayObject);
    int ndims = PyArray_NDIM(reinterpret_cast<PyArrayObject*>(pyArrayObject));
    if (ndims != 2)
    {
        throw except::Exception(Ctxt(
                    "Numpy array has dimensions different than 2"));
        return 0;
    }
    return PyArray_DIMS(reinterpret_cast<PyArrayObject*>(pyArrayObject));
}

types::RowCol<size_t> getDimensionsRC(PyObject* pyArrayObject)
{
   const npy_intp* dims = getDimensions(pyArrayObject);
   return types::RowCol<size_t>(dims[0], dims[1]);
}

void verifyObjectsAreOfSameDimensions(PyObject* lhs, 
                                                  PyObject* rhs)
{
    if(getDimensionsRC(lhs) != getDimensionsRC(rhs))
    {
        throw except::Exception(Ctxt(
                    "Numpy arrays are of differing dimensions"));
    }
}

void createOrVerify(PyObject*& pyObject,
                                int typeNum, 
                                const types::RowCol<size_t>& dims)
{
    if (pyObject == Py_None) // none passed in-- so create new
    {
        npy_intp odims[2] = {dims.row, dims.col};
        pyObject = PyArray_SimpleNew(2, odims, typeNum);
    }
    else
    {
        verifyArrayType(pyObject, typeNum);
        const npy_intp* const outdims = getDimensions(pyObject);
        if (outdims[0] != dims.row  || outdims[1] != dims.col)
        {
            throw except::Exception(Ctxt(
                        "Desired array does not match required row, cols"));
        }
    }
}

void prepareInputAndOutputArray(PyObject* pyInObject, 
                                            PyObject*& pyOutObject, 
                                            int inputTypeNum, 
                                            int outputTypeNum, 
                                            types::RowCol<size_t> dims)
{
    verifyArrayType(pyInObject, inputTypeNum);
    createOrVerify(pyOutObject, outputTypeNum, dims);
}

void prepareInputAndOutputArray(PyObject* pyInObject,
                                            PyObject*& pyOutObject,
                                            int inputTypeNum,
                                            int outputTypeNum)
{
    prepareInputAndOutputArray(pyInObject,
                               pyOutObject,
                               inputTypeNum,
                               outputTypeNum,
                               getDimensionsRC(pyInObject));
}
                     
}

