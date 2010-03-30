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

#ifndef __MATRIX_INTERFACE_H__
#define __MATRIX_INTERFACE_H__

#include "linear/lite/Cast.h"
#include "linear/lite/VectorInterface.h"

/*!
 *  \file MatrixInterface.h
 *  \brief The interface for a matrix driver class
 *
 */

namespace linear
{
	namespace lite
	{
	/*!
	 *  \class MatrixInterface
	 *  \brief Base class for matrix operations
	 *
	 *  MatrixInterface is an interface, and partial implementation for performing
	 *  two-dimensional matrix operations.  Derived classes can then be implemented
	 *  using various math libraries as drivers to implement the required methods.
	 *
	 *  In order to get transparency of implementation libraries, the matrix interface
	 *  API refers always to base classes in its function declarations.  For example,
	 *  a matrix-vector multiply operation will refer to a VectorInterface as the parameter
	 *  argument, and will produce a VectorInterface as a return type.  There are several 
	 *  implications for this type of behavior.  It is up to the implementor whether
	 *  an implementation of this interface will interop with other implementations.  Typically,
	 *  different math library bindings will not be written to interop (this is the case
	 *  for all current implementations), even though the interface "allows" it.  All existing
	 *  derived classes immediately cast parameter arguments from the base interface to derived
	 *  interfaces from the same math library.  By default, this cast is a dynamic_cast,
	 *  which are runtime cast attempts.  On failure, these casts throw std::bad_cast exceptions.
	 *  Library users have the option of turning off dynamic checks by defining LINEAR_LITE_NO_RTTI_CHECK
	 *  at compile time.  This will turn all casts into static_cast.  This method will be preferred for not
	 *  incurring runtime check performance hits, but should only be used when their is only one math library
	 *  supplying these calls.  Another implication for the previous example is that the VectorInterface is
	 *  known by the MatrixInterface (in order to have a mat-vec mult op).  The opposite is not true (and is not
	 *  included to prevent cross-dependencies).
	 *
	 *  There is some flexibility to the implementor of the matrix interface.  Some classes may use additional
	 *  templating to refer to underlying storage types abstractly as well.  This may prove handy for a derived
	 *  implementation using a flexible enough template library to treat dense and sparse matrices identically, for
	 *  example.
	 *
	 *  The philosophy behind the MatrixInterface method specification is to declare only the minimum number of
	 *  functions necessary to do basic linear math.  Additionally, all operations are declared internal-const
	 *  wherever possible.  There is a handle to keep abstraction from the MatrixInterface directly, called simply
	 *  "Matrix."  This class is the preferred mechanism for matrix math.
	 *
	 *  While nothing prevents library users from creating a derived class object statically, and making subsequent
	 *  calls, doing so will prove clumsy, in that all return values for structures are pointers and are new allocated
	 *  within this class.  Instead the preferred method for accessing the matrix set of methods is by instantiating
	 *  a proxy Matrix object using a MatrixCreator:
	 *
     *  \code
	      Matrix<double> mx = MatrixCreatorMTL<double>().newIdentityMatrix(4); // produce a 4x4 ident matrix 
          // make proxy calls which call underlying MatrixInterface functions
		  mx.scale(0.62);
		  mx(3, 3) = 186.4;
	 *  \endcode
	 *
	 *  
	 */

	template<typename T> class MatrixInterface
	{
	public:
		//! Default construct
		MatrixInterface() {}
		//! Row column initialization (does nothing)
		MatrixInterface(int rows, int cols) {}
		
		//! Virtual destructor
		virtual ~MatrixInterface() {}

		/*!
		 *  Initialize matrix storage to the size specified
		 *  \param rows The number of rows in the matrix
		 *  \param cols The number of cols in the matrix
		 */
		virtual void init(int rows, int cols) = 0;

		/*!
		 * Operation to retrieve number of rows in this matrix
		 * \return number of rows in matrix
		 */
		virtual int rows() const = 0;

		/*!
		 * Operation to retrieve number of columns in this matrix
		 * \return number of columns in matrix
		 */
		virtual int cols() const = 0;

		/*!
		 * Const operation to retrieve element at i,j in this matrix
		 * \return element at i,j (copy)
		 */
		virtual T get(int i, int j) const = 0;
		
		/*!
		 * Non-const operation to retrieve element at i,j in this matrix.
		 * 
		 * \return element at i,j (ref)
		 */
		virtual T& get(int i, int j)
                {
                    throw except::Exception(Ctxt("Not implemented, use set()!"));
                }

                /*!
                 *  Copy the data from the matrix (const) to the given pointer.
                 *  The pointer must be pre-allocated.
                 *  \param dataPtr The pointer to copy the matrix's data to
                 */
                void get(T* dataPtr) const
                {
                   int numRows = rows();
                   int numCols = cols();

                   for(int i=0; i<numRows; i++)
                   {
                      for(int j=0; j<numCols; j++)
                      {
                         dataPtr[i*numCols + j] = get(i, j);
                      }
                   }
                }


		/*!
		 *  Rather than have a (non-const) reference operator()
		 *  overload, we provide a set method.  This is because
		 *  it appears that uBlas, and perhaps MTL do not actually
		 *  offer a sparse matrix reference to the actual raw type.
		 *  Instead, they provide an encapsulating object which can
		 *  be used in the expected manner.  Since this type cannot
		 *  be offered in the MatrixInterface, we elect to hide it
		 *  entirely from the user and not allow a non-const accessor
		 *  get() or operator() method.  Hopefully, a work-around can
		 *  be devised in the near future.
		 *
		 *  \param i row index
		 *  \param j column index
		 *  \param elem A value to copy into the element in the specified position
		 */
		virtual void set(int i, int j, const T& elem) = 0;

                /*!
                 *  Copy the data from the given pointer to the matrix.
                 *  \param numRows The number of rows in the data
                 *  \param numCols The number of columns in the data
                 *  \param dataPtr The pointer to copy into the matrix
                 */
		virtual void set(int numRows, int numCols, const T* dataPtr)
                {
                   //check the size 
                   if(rows() != numRows || cols() != numCols)
                   {
                      init(numRows, numCols);
                   }

                   for (int i = 0; i < numRows; i++)
                   {
                      for (int j = 0; j < numCols; j++)
                         set(i, j, dataPtr[i*numCols + j]);
                   }
                }

		/*!
		 * Clone (deep-copy) this matrix
		 * \return a cloned matrix
		 */
		virtual MatrixInterface* clone() const = 0;

		/*!
		 * Const operator to retrieve element at i,j in this matrix
		 * \return element at i,j (copy)
		 */
		T operator()(int i, int j) const { return get(i, j); }
		
		/*!
		 * Non-const operator to retrieve element at i,j in this matrix
		 * \return element at i,j (ref)
		 */
		//T& operator()(int i, int j) { return get(i, j); }

		/*!
		 *  Create and return a matrix which is the inverse of
		 *  this matrix.  Internal matrix is not modified.
		 *  \return A dynamically allocated matrix
		 */
		virtual MatrixInterface* transpose() const = 0;
		
		/*!
		 *  Create and return a matrix which is the inverse of
		 *  this matrix.  Internal matrix is not modified.
		 *  Currently, behavior is unspecified if no inverse exists.
		 *
		 *  \return A dynamically allocated matrix
		 */
		virtual MatrixInterface* inverse() const = 0;
		
		/*!
		 *  Create and return a matrix which is the sum of
		 *  this matrix and the parameter mx.  Internal matrix 
		 *  is not modified.  
		 *  \param mx A matrix to add to this
		 *  \return A dynamically allocated matrix
		 */
		virtual MatrixInterface* add(const MatrixInterface& mx) const = 0; 
		/*!
		 *  Create and return a matrix which is the product of
		 *  this matrix and the parameter mx.  Internal matrix 
		 *  is not modified.  
		 *
		 *  \param mx A matrix to multiply to this
		 *  \return A dynamically allocated matrix
		 */
		virtual MatrixInterface* multiply(const MatrixInterface& mx) const = 0;

		/*!
		 *  Create and return a matrix which is the product of this matrix
		 *  and a scale value.  Internal matrix is not modified.
		 *
		 *  \param scalar A scalar to multiply this matrix by
		 *  \return A dynamically allocated matrix
		 */
		virtual MatrixInterface* multiply(T scalar) const = 0;

		/*!
		 *  Create and return a matrix which is the product of this matrix
		 *  and a vector.  Internal matrix is not modified.
		 *
		 *  \param scalar A scalar to multiply this matrix by
		 *  \return A dynamically allocated matrix
		 */
		virtual VectorInterface<T>* multiply(const VectorInterface<T>* vec) const = 0;

		/*!
		 *  This method is essentially the same as the scalar multiply, except that
		 *  it mutates an existing matrix.
		 *
		 *  \param scalar A scalar to multiply this matrix by
		 */
		virtual void scale(T scalar) = 0;

		/*!
		 *  Returns the determinant of this matrix
		 *
		 *  \return A scalar value
		 */
		virtual T determinant() const = 0;

		/*!
		 *  Returns the trace of this matrix.  This function has 
		 *  a stand-in default implementation, since not many libraries
		 *  seem to implement this trivial function.  It may be overridden
		 *  in derived classes.
		 *
		 *  \return A scalar value
		 */
		virtual T trace() const
		{
                    T tr(0);
                    if (rows() != cols())
                        throw except::Exception(Ctxt("Require a square matrix"));
                    for (int i = 0; i < rows(); i++)
                        tr += get(i, i);
                    return tr;
		}
		
		/*!
		 *   Native object print routine.  This function exposes
		 *   whatever built-in print operation exists in the implmentation
		 *   math library.
		 */
		virtual void print() const = 0;
	};

	/*!
	 *  \class MatrixCreatorInterface
	 *  \brief Interface using factory creation pattern to produce an actual matrix
	 *
	 *  This class is a pure interface for the creation of matrices.  Implementations
	 *  for specific math packages will need to derive this class with a version that
	 *  produces their sub-classed MatrixInterface.  
	 *  
	 *  The MatrixCreatorPartial may be sub-classed, or this may be derived directly
	 *  to fulfill the interface requirements.
	 */
	template<typename T> class MatrixCreatorInterface
	{
	public:

		//! Default constructor
		MatrixCreatorInterface() {}

		//! Virtual destructor
		virtual ~MatrixCreatorInterface() {}

		/*!
		 *  Create a new matrix with number of rows and columns specified.
		 *  Whether or not contained values are initialized is undefined.
		 *  \return A dynamically allocated Matrix of type T
		 */
		virtual MatrixInterface<T>* newMatrix(int rows, int cols) const = 0;

	    /*!
		 *  Create a new matrix with number of rows and columns specified,
		 *  with all elements initialized to the data values specified
		 *  in the input parameters
		 *
		 *  \param rows The number of rows desired in the matrix
		 *  \param cols The number of cols desired in the matrix
		 *  \param dataPtr The data values to intialize each element with
		 *  \return A dynamically allocated Matrix of type T
		 */
		virtual MatrixInterface<T>* newMatrix(int rows, int cols, const T* dataPtr) const = 0;

	    /*!
		 *  Create a new matrix with number of rows and columns specified,
		 *	with all elements initialized to the constant value specified
		 *  in the input parameters
		 *
		 *  \param rows The number of rows desired in the matrix
		 *  \param cols The number of cols desired in the matrix
		 *  \param constantValue A value to intialize each element with
		 *  \return A dynamically allocated Matrix of type T
		 */
		virtual MatrixInterface<T>* newConstantMatrix(int rows, int cols, T constantValue) const = 0;

		/*!
		 *  Create a new square identity matrix with cols and rows specified as
		 *  single argument
		 *  
		 *  \param d The number of rows and cols of this square matrix.
		 *  \return A dynamically allocated Matrix of type T
		 */
		virtual MatrixInterface<T>* newIdentityMatrix(int d) const = 0;
		
	};

	/*!
	 *  \class MatrixCreatorPartial
	 *  \brief pseudo-implementation for factory creation
	 *
	 *  This class provides non-optimized creation routines for the constant and
	 *  identity matrix methods.  It does not provide the basic creator function
	 */
	template<typename T> class MatrixCreatorPartial : public MatrixCreatorInterface<T>
	{

	public:
		//! Default constructor
		MatrixCreatorPartial() {}
		//! Virtual destructor
		virtual ~MatrixCreatorPartial() {}
		
		virtual MatrixInterface<T>* newMatrix(int rows, int cols) const = 0;
		/*!
		 *  Creates a new matrix and initializes it with the given data in two steps.
		 *  \param rows The number of rows desired in the matrix
		 *  \param cols The number of cols desired in the matrix
		 *  \param dataPtr The data to use to initialize the matrix
		 *  \return A dynamically allocated Matrix of type T
		 */
		virtual MatrixInterface<T>* newMatrix(int rows, int cols, const T* dataPtr) const
		{
			MatrixInterface<T>* mat = this->newMatrix(rows, cols);
                        mat->set(rows, cols, dataPtr);
			return mat;
		}
		
		/*!
		 *  Creates a new constant matrix in two steps.  This method is not efficient and should
		 *  be used only if a constant matrix creation routine is not offered in the implementing
		 *  math library
		 *  \param rows The number of rows desired in the matrix
		 *  \param cols The number of cols desired in the matrix
		 *  \return A dynamically allocated Matrix of type T
		 */
		virtual MatrixInterface<T>* newConstantMatrix(int rows, int cols, T constantValue) const
		{
			MatrixInterface<T>* mat = this->newMatrix(rows, cols);
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
					mat->set(i, j, constantValue);
			}
			return mat;
		}

		/*!
		 *  Creates a new constant matrix in two steps (currently three indirectly).  
		 *  This method is not efficient and should
		 *  be used only if an identity matrix creation routine is not offered in the implementing
		 *  math library
		 *  \param rows The number of rows desired in the matrix
		 *  \param cols The number of cols desired in the matrix
		 *  \return A dynamically allocated Matrix of type T
		 */
		virtual MatrixInterface<T>* newIdentityMatrix(int d) const
		{
			MatrixInterface<T>* mat = newConstantMatrix(d, d, 0);
			for (int i = 0; i < d; i++)
			{
				T one(1);
				mat->set(i, i, one);
			}
			return mat;	
		}

	};
}
}

#endif
