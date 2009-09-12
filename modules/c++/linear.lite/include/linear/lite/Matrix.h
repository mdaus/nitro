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

#ifndef __LINEAR_LITE_MATRIX_H__
#define __LINEAR_LITE_MATRIX_H__

#include "linear/lite/Vector.h"
#include "linear/lite/MatrixInterface.h"
#include <memory>

/*!
 *  \file Matrix.h
 *  \brief Matrix and matrix-related routines and classes for this module
 *
 */

namespace linear
{
	namespace lite
	{


   /*!
	*  \class Matrix
	*  \brief Encapsulation type for a matrix
	*
	*  The Matrix class is a handle to some underlying driver which implements the
	*  MatrixInterface.  The driver is managed by pointer, and is required to be
	*  created externally using some factory creation pattern, or by somehow
	*  initializing from an existing matrix interface.
	*
	*
	*/
	template<typename T> class Matrix
	{
	public:

		/*!
		 *  Constructor from a MatrixInterface.
		 *  This operation takes ownership of the pointer.
		 *   
		 *  \param newMatrix A matrix interface object to contain
		 */
		Matrix(MatrixInterface<T>* newMatrix)
		{
			mxi.reset(newMatrix);
		}

		/*!
		 *  Copy constructor.  Does a clone operation on the existing Matrix's
		 *  interface to produce a new deep copy.
		 *
		 *  \param mx Existing matrix to copy from
		 */
		Matrix(const Matrix& mx)
		{
			mxi.reset(mx.mxi->clone());
		}
		
		/*!
		 *  This is an assignment operator for a new matrix.
		 *  It uses the underlying Matrix's interface clone()
		 *  function to retrieve a deep copy of the pointer.
		 *
		 *  \param mx The matrix to initialize from
		 *  \return This
		 */
		Matrix& operator=(const Matrix& mx)
		{
			// check self assign
			if (&mx != this)
			{
				mxi.reset(mx.mxi->clone());
			}
			return *this;
		}

		/*!
		 *  Assignment from a Matrix interface pointer, which this will
		 *  contain.  This is used for implicit type conversion from methods
		 *  within the MatrixInterface, which return new-allocated pointers
		 *  to MatrixInterface objects.
		 *
		 *  \param newMatrix A matrix to contain
		 */
		Matrix& operator=(MatrixInterface<T>* newMatrix)
		{
			mxi.reset(newMatrix);
                        return *this;
		}

                /*!
                 *  This function augments the current matrix with the given
                 *  vector.  The vector values are added in a new column.
                 *  The size of the vector must match the number of rows
                 *  in the matrix.
                 *
                 *  \param vec The vector to augment this matrix with
                 */
                void appendColumnVector(Vector<T>& vec)
                {
                   int origCols = this->cols();
                   int numCols = origCols + 1;
                   int numRows = this->rows();

                   if(numRows != vec.size())
                   {
                      //sizes don't match - throw exception
                      throw except::Exception(Ctxt("Matrix::appendColumnVector: ERROR - sizes don't match"));
                   }

                   T* ptr = new T[numRows * numCols];

                   for(int i=0; i<numRows; i++)
                   { 
                      for(int j=0; j<origCols; j++)
                      {
                         ptr[i*numCols + j] = this->get(i, j);
                      }
                      ptr[i*numCols + origCols] = vec.get(i);
                   }

                   set(numRows, numCols, ptr);
                   delete ptr;
                }

                /*!
                 *  This function augments the current matrix with the given
                 *  vector.  The vector values are added in a new row.
                 *  The size of the vector must match the number of columns 
                 *  in the matrix.
                 *
                 *  \param vec The vector to augment this matrix with
                 */
                void appendRowVector(Vector<T>& vec)
                {
                   int origRows = this->rows();
                   int numRows = origRows + 1;
                   int numCols = this->cols();

                   if(numCols != vec.size())
                   {
                      //sizes don't match - throw exception
                      throw except::Exception(Ctxt("Matrix::appendRowVector: ERROR - sizes don't match"));
                   }

                   T* ptr = new T[numRows * numCols];

                   for(int i=0; i<origRows; i++)
                   { 
                      for(int j=0; j<numCols; j++)
                      {
                         ptr[i*numCols + j] = this->get(i, j);
                      }
                   }
                   for(int j=0; j<numCols; j++)
                   {
                      ptr[origRows*numCols + j] = vec.get(j);
                   }

                   set(numRows, numCols, ptr);
                   delete ptr;
                }

                /*!
                 *  This function augments the current matrix with the given
                 *  matrix.  The matrix values are added in new columns.
                 *  The the number of rows in both matrices must match.
                 *
                 *  \param mx The matrix to augment this matrix with
                 */
                void appendColumns(Matrix& mx)
                {
                   int origCols = this->cols();
                   int numCols = origCols + mx.cols();
                   int numRows = this->rows();

                   if(numRows != mx.rows())
                   {
                      //sizes don't match - throw exception
                      throw except::Exception(Ctxt("Matrix::appendColumns: ERROR - sizes don't match"));
                   }

                   T* ptr = new T[numRows * numCols];

                   for(int i=0; i<numRows; i++)
                   { 
                      for(int j=0; j<origCols; j++)
                      {
                         ptr[i*numCols + j] = this->get(i, j);
                      }
                      for(int j=origCols; j<numCols; j++)
                      {
                         ptr[i*numCols + j] = mx.get(i, j-origCols);
                      }
                   }

                   set(numRows, numCols, ptr);
                   delete ptr;
                }

                /*!
                 *  This function augments the current matrix with the given
                 *  matrix.  The matrix values are added in new rows.
                 *  The the number of columns in both matrices must match.
                 *
                 *  \param mx The matrix to augment this matrix with
                 */
                void appendRows(Matrix& mx)
                {
                   int origRows = this->rows();
                   int numRows = origRows + mx.rows();
                   int numCols = this->cols();

                   if(numCols != mx.cols())
                   {
                      //sizes don't match - throw exception
                      throw except::Exception(Ctxt("Matrix::appendRows: ERROR - sizes don't match"));
                   }

                   T* ptr = new T[numRows * numCols];

                   for(int i=0; i<origRows; i++)
                   { 
                      for(int j=0; j<numCols; j++)
                      {
                         ptr[i*numCols + j] = this->get(i, j);
                      }
                   }
                   for(int i=origRows; i<numRows; i++)
                   { 
                      for(int j=0; j<numCols; j++)
                      {
                         ptr[i*numCols + j] = mx.get(i-origRows, j);
                      }
                   }

                   set(numRows, numCols, ptr);
                   delete ptr;
                }

                /*!
                 *  This function selects a contiguous section of columns 
                 *  from this matrix and returns a new matrix containing 
                 *  that data.
                 *
                 *  \param startCol The index of the first column to copy from
                 *  \param endCol The index of the last column to copy from
                 */
                void getColumns(int startCol, int endCol, Matrix<T>& newMat)
                {
                   int numRows = this->rows();
                   int numCols = endCol-startCol+1;

                   T* ptr = new T[numRows * numCols];

                   for(int i=0; i<numRows; i++)
                   { 
                      for(int j=0; j<numCols; j++)
                      {
                         ptr[i*numCols + j] = get(i, j+startCol);
                      }
                   }

                   newMat.set(numRows, numCols, ptr);
                   delete ptr;
                }

                /*!
                 *  This function copies a column of data from this matrix
                 *  into the given vector.
                 *
                 *  \param col The index of the column to copy
                 *  \param vec The vector to copy the data to
                 */
                void getColumnVector(int col, Vector<T>& vec)
                {
                   int numRows = this->rows();

                   T* ptr = new T[numRows];

                   for(int i=0; i<numRows; i++)
                   { 
                      ptr[i] = get(i, col);
                   }

                   vec.set(numRows, ptr);
                   delete ptr;
                }

                /*!
                 *  This function copies a row of data from this matrix
                 *  into the given vector.
                 *
                 *  \param row The index of the row to copy
                 *  \param vec The vector to copy the data to
                 */
                void getRowVector(int row, Vector<T>& vec)
                {
                   int numCols = this->cols();

                   T* ptr = new T[numCols];

                   for(int i=0; i<numCols; i++)
                   { 
                      ptr[i] = get(row, i);
                   }

                   vec.set(numCols, ptr);
                   delete ptr;
                }

                /*!
                 *  This function copies the data from the given vector
                 *  into a column in this matrix.  The size of the vector
                 *  must match the number of rows in the matrix.
                 *
                 *  \param col The index of the column to copy to
                 *  \param vec The vector to copy the data from
                 */
                void setColumnVector(int col, Vector<T>& vec)
                {
                   int numRows = this->rows();

                   if(numRows != vec.size())
                   {
                      //sizes don't match - throw exception
                      throw except::Exception(Ctxt("Matrix::setColumnVector: ERROR - sizes don't match"));
                   }

                   for(int i=0; i<numRows; i++)
                   { 
                      set(i, col, vec.get(i));
                   }
                }

                /*!
                 *  This function copies the data from the given vector
                 *  into a row in this matrix.  The size of the vector
                 *  must match the number of columns in the matrix.
                 *
                 *  \param row The index of the row to copy to
                 *  \param vec The vector to copy the data from
                 */
                void setRowVector(int row, Vector<T>& vec)
                {
                   int numCols = this->cols();

                   if(numCols != vec.size())
                   {
                      //sizes don't match - throw exception
                      throw except::Exception(Ctxt("Matrix::setRowVector: ERROR - sizes don't match"));
                   }

                   for(int i=0; i<numCols; i++)
                   { 
                      set(row, i, vec.get(i));
                   }
                }

		/*!
		 *  Create and return the transpose of this.
		 *  \sa MatrixInterface::transpose
		 *
		 *  \return A new transpose matrix
		 */
		MatrixInterface<T>* transpose() const
		{
			
			MatrixInterface<T>* tr = mxi->transpose();
			return tr;
		}
	
		/*!
		 *  Create and return the transpose of this.
		 *  \sa MatrixInterface::inverse
		 *
		 *  \return A new transpose matrix
		 */
		MatrixInterface<T>* inverse() const 
		{
			return mxi->inverse();
		}

		/*!
		 *  Scale this matrix by the value specified.
		 *  \sa MatrixInterface::inverse
		 *  
		 *  \param scalar The value to scale by
		 *  \return A new transpose matrix
		 */
		void scale(T scalar)
		{
			mxi->scale(scalar);
		}

		/*!
		 *  Matrix addition.
		 *  \sa MatrixInterface::add
		 *
		 *  \param mx The matrix to add to this
		 *  \return A new summation of this and the argument
		 */
		MatrixInterface<T>* add(const Matrix<T>& mx) const
		{
			return mxi->add(*(mx.mxi));
		}

		/*!
		 *  Matrix multiplication.
		 *  \sa MatrixInterface::multiply
		 *
		 *  \param mx The matrix to add to this
		 *  \return A new summation of this and the argument
		 */
		MatrixInterface<T>* multiply(const Matrix& mx) const
		{
			return mxi->multiply(*(mx.mxi));
		}

		/*!
		 *  Matrix multiplication.
		 *  \sa MatrixInterface::multiply
		 *
		 *  \param mx The matrix to add to this
		 *  \return A new summation of this and the argument
		 */
		VectorInterface<T>* multiply(const Vector<T>& vec) const
		{
		  
			return mxi->multiply((const VectorInterface<T>*)vec.veci.get());
		}
		/*!
		 *  Multiply this by a scalar and return the product.
		 *  \sa MatrixInterface::multiply
		 *  \sa MatrixInterface::scale
		 *
		 *  \param mx The matrix to add to this
		 *  \return A new summation of this and the argument
		 *
		 */
		MatrixInterface<T>* multiply(T scalar) const
		{
			return mxi->multiply(scalar);
		}
		/*!
		 *  Return the number of rows in this matrix
		 */
		int rows() const { return mxi->rows(); }
		/*!
		 *  Return the number of columns in this matrix
		 */
		int cols() const { return mxi->cols(); }

                /*!
                 *  Copy the data from the given pointer to the matrix
                 *  \param numRows The number of rows in the data
                 *  \param numCols The number of columns in the data
                 *  \param dataPtr The pointer to copy into the matrix
                 */
                virtual void set(int numRows, int numCols, const T* dataPtr)
                {
                   mxi->set(numRows, numCols, dataPtr);
                }

		void set(int i, int j, const T& elem) { mxi->set(i, j, elem); }

                /*!
                 * Clone (deep-copy) this matrix
                 * \return a cloned matrix
                 */
                virtual MatrixInterface<T>* clone()
                {
                   return mxi->clone();
                }

		/*!
		 *  Const-accessor for element at ith row, jth column
		 *
		 *  \sa MatrixInterface::get
		 *
		 *  \param i The ith row
		 *  \param j The jth column
		 *  \return A copy of the element a ith row, jth column
		 */
		T get(int i, int j) const { return mxi->get(i, j); }

		/*!
		 *  Reference-accessor for element at ith row, jth column
		 *
		 *  \sa MatrixInterface::get
		 *
		 *  \param i The ith row
		 *  \param j The jth column
		 *  \return A reference to the element a ith row, jth column
		 */
		T& get(int i, int j) { return mxi->get(i, j); }

                /*!
                 *  Copy the data from the matrix (const) to the given pointer.
                 *  The pointer must be pre-allocated.
                 *  \param dataPtr The pointer to copy the matrix's data to
                 */
                void get(T* dataPtr) const
                {
                   mxi->get(dataPtr);
                }

		/*!
		 *  Const-operator for element at ith row, jth column
		 *
		 *  \sa MatrixInterface::get
		 *
		 *  \param i The ith row
		 *  \param j The jth column
		 *  \return A copy of the element a ith row, jth column
		 */
		T operator()(int i, int j) const { return get(i, j); }
		
		/*!
		 *  Reference-operator for element at ith row, jth column
		 *
		 *  \sa MatrixInterface::get
		 *
		 *  \param i The ith row
		 *  \param j The jth column
		 *  \return A reference to the element a ith row, jth column
		 */
		T& operator()(int i, int j) { return get(i, j); }

		/*!
		 *  Retrieve the determinant of a matrix
		 *  \sa MatrixInterface::determinant
		 *
		 *  \return The determinant
		 */
		T determinant() const { return mxi->determinant(); }
		
		/*!
		 *  Retrieve the trace of a matrix
		 *  \sa MatrixInterface::trace
		 *
		 *  \return The trace
		 */
		T trace() const { return mxi->trace(); }
		
		/*!
		 *  Show the matrix representation in native form.
		 *
		 */
		void print() const { mxi->print(); }
	private:
		std::auto_ptr< MatrixInterface<T> > mxi;
		
		Matrix() {}

	};
	
	

	template<typename T> Matrix<T>
	  operator*(const Matrix<T>& m, T scalar)
	{
	    return m.multiply(scalar);
	}


	template<typename T> Matrix<T>
	  operator*(const Matrix<T>& m1, const Matrix<T>& m2)
	{
	    return m1.multiply(m2);
	}

	template<typename T> Vector<T>
	  operator*(const Matrix<T>& m, const Vector<T>& v)
	{
	    return m.multiply(v);
	}


	template<typename T> Matrix<T>
	  operator*(T scalar, const Matrix<T>& m)
	{
	    return m.multiply(scalar);
	}

	template<typename T> Matrix<T> operator+(const Matrix<T>& m1, 
						 const Matrix<T>& m2)
	{
	    return m1.add(m2);
	}



	template<typename T> Matrix<T> operator-(const Matrix<T>& m1, 
						 const Matrix<T>& m2)
        {
	    Matrix<T> negative(m2);
	    negative.scale(-1);
	    return m1.add(negative);
	}
	
	template<typename T> std::ostream& operator<<(std::ostream& os,
						      const Matrix<T>& m)
	{

	    
	    int i, j;

	    for (i = 0; i < m.rows(); ++i)
	    {
		for (j = 0; j < m.cols() - 1; ++j)
		{
		    os << m(i, j) << " "; 
		}
		os << m(i, j) << std::endl;

		
		    
	    }
		    

	    return os;
	}


	}
}


#endif
