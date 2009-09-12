#ifndef __MATRIX_MTL_H__
#define __MATRIX_MTL_H__

#include "linear/lite/MatrixInterface.h"
#include "linear/lite/mtl/VectorMTL.h" // mat-vec mult


#include <cstdio>
#include <cmath>
#include <iostream>
#include "mtl/mtl.h"
#include "mtl/utils.h"
#include "mtl/linalg_vec.h"
#include "mtl/lu.h"

/*!
 *  \file MatrixMTL.h
 *  \brief An implementation driver for the Matrix Template Library (MTL)
 *
 */

namespace linear
{
	namespace lite
	{
	namespace mtl
	{

	/*!
	 *  \class MatrixMTL
	 *  \brief Matrix implementation using the Matrix Template Library
	 *
	 *  This class binds our MatrixInterface to the Matrix Template Library, a commonly
	 *  used, high-performance C++ Matrix library.  The default parameter in the template
	 *  Mx_T allows the type to propogate without having to have a concrete name in the class.
	 *  It also serves the purpose of allowing explicit alternate storage available within the MTL.
	 *  This class is tentatively final (through lack of virtual declarations).
	 *
	 */
	template<typename T, typename Mx_T = mtl::matrix<typename T>::type> class MatrixMTL : public MatrixInterface<T>
	{
	public:
		//!  Default constructor
	    MatrixMTL() {}

		/*!
		 *  Construct a matrix with rows and columns specified.
		 *
		 *  \param rows The number of rows to create
		 *  \param col The number of columns to create
		 */
	    MatrixMTL(int rows, int cols)
	    {
			internal = Mx_T(rows, cols);
	    }

		/*!
		 *  Construct a matrix from a native MTL type.
		 *  This matrix's elements are copied to the internal storage
		 *
		 *  \param mx The native MTL matrix to copy
		 */
		MatrixMTL(const Mx_T& mx)
		{
			internal = Mx_T(mx.nrows(), mx.ncols());
			mtl::copy(mx, internal);
		}

		//!  Destructor
		~MatrixMTL() {}

		/*!
		 *  Return a base pointer, pointing at a new MTL matrix driver that
		 *  is a deep copy of this
		 *
		 *  \return A deep copy matrix
		 */
		MatrixInterface<T>* clone() const
		{
			return new MatrixMTL<T, Mx_T>(this->internal);
		}

		/*!
		 *  Initialize the matrix to have the number of rows and columns
		 *  specified in the parameter list.
		 *
		 *  \param rows The number of rows this matrix should have
		 *  \param cols The number of columns this matrix should have
		 */
		void init(int rows, int cols) { internal = Mx_T(rows, cols); }

		/*!
		 *  Get the number of rows in this matrix
		 */
	    int rows() const { return internal.nrows(); }

		/*!
		 *  Get the number of columns in this matrix
		 */
	    int cols() const { return internal.ncols(); }

		/*!
		 *  Const-accessor for element at (i, j).
		 *
		 *  \param i The ith row
		 *  \param j The jth col
		 *  \reurn A copy of the element at (i, j)
		 */
	    T get(int i, int j) const { return internal(i, j); }

		/*!
		 *  non-const-accessor for element at (i, j).
		 *
		 *  \param i The ith row
		 *  \param j The jth col
		 *  \reurn A reference to the element at (i, j)
		 */
	    //T& get(int i, int j) { return internal(i, j); }

		/*!
		 *  Explicit set operation in place of non-constant reference
		 *  accessor.
		 *  \sa MatrixInterface::set
		 *
		 *  \param i row index
		 *  \param j col index
		 *  \param elem new element value
		 */
		void set(int i, int j, const T& elem) { internal(i, j) = elem; }
		/*!
		 *  Create a new MTL matrix that is the transpose of this.  The
		 *  new matrix will have the same storage as this.
		 *
		 *  \return A new MTL matrix transpose
		 */
	    MatrixInterface<T>* transpose() const
	    {
			MatrixMTL<T, Mx_T> *mx = new MatrixMTL<T, Mx_T>(internal.nrows(),
												internal.ncols());

			mtl::transpose(this->internal, mx->internal);
			return mx;
	    }

		/*!
		 *  Create a new MTL matrix that is the inverse of this.  The
		 *  new matrix will have the same storage as this.  If no inverse
		 *  exists, results may be undefined
		 *
		 *  \return A new MTL matrix inverse
		 */
	    MatrixInterface<T>* inverse() const
	    {
			MatrixMTL<T, Mx_T> *inv = new MatrixMTL<T, Mx_T>(internal.nrows(),
											     internal.ncols());
			mtl::dense1D<size_t> pvec(internal.nrows());
			Mx_T lu(internal.nrows(), internal.ncols());
			mtl::copy(this->internal, lu);
			mtl::lu_factor<Mx_T, mtl::dense1D<size_t> >(lu, pvec);
			mtl::lu_inverse(lu, pvec, inv->internal);
			return inv;
	    }

		/*!
		 *  Create a new matrix which is the sum of this and its argument.
		 *  This method requires that the passed argument be of like type
		 *  to this (i.e., both MTL, same 'T', same storage type 'Mx_T' )
		 *
		 *  \param mx The argument to add
		 *  \return A new matrix with the same storage as this
		 */
		MatrixInterface<T>* add(const MatrixInterface<T>& mx) const
	    {
			//dynamic_cast<const MatrixMTL<T, Mx_T >& >(mx)
			const MatrixMTL<T, Mx_T>& reint = down_cast< const MatrixMTL<T, Mx_T>&, const MatrixInterface<T>& >(mx);
			Mx_T prod(internal);
			mtl::add(reint.internal, prod);
			return new MatrixMTL<T, Mx_T>(prod);
	    }
		/*!
		 *  Create a new matrix which is the product of this and its argument.
		 *  This method requires that the passed argument be of like type
		 *  to this (i.e., both MTL, same 'T', same storage type 'Mx_T' )
		 *
		 *  \param mx The argument to multiply
		 *  \return A new matrix with the same storage as this*
		 */
	    MatrixInterface<T>* multiply(const MatrixInterface<T>& mx) const
	    {
			//MatrixInterface<typename T> *prod = new MatrixMTL<typename T>(this->internal.nrows(),
			//								      mx.internal.ncols());
			//const MatrixMTL<T>& reint = LINALG_CAST(const MatrixMTL<T>&, mx);
			const MatrixMTL<T, Mx_T>& reint = down_cast< const MatrixMTL<T, Mx_T>&, const MatrixInterface<T>& >(mx);
			MatrixMTL<T, Mx_T> *prod = new MatrixMTL<T, Mx_T>(this->internal.nrows(),
												  reint.internal.ncols());
			mtl::mult(this->internal, reint.internal, prod->internal);
			return prod;
	    }

		/*!
		 *  Create a new vector which is the product of this and its argument.
		 *  This method requires that the passed argument be of like type
		 *  to the argument parameter (i.e., both MTL, same 'T',
		 *  same storage type 'Storage_T' )
		 *
		 *  \sa VectorUBlas
		 *
		 *  \param mx The argument to multiply
		 *  \return A new matrix with the same storage as this*
		 */
		VectorInterface<T>* multiply(const VectorInterface<T>* vec) const
		{
			const VectorMTL<T>* reint = down_cast<const VectorMTL<T>*, const VectorInterface<T>* >(vec);
			assert(reint); // because its a pointer, not a reference
			assert(cols() == reint->size());
			VectorMTL<T>* prod = new VectorMTL<T>(rows());
			mtl::mult(internal, reint->native(), prod->native());
			return prod;
		}
		/*!
		 *  Multiply this by a scalar and return the result matrix
		 *
		 *  \param scalar The value to scale all elements by
		 *
		 *  \return A new matrix of like type to this
		 */
	    MatrixInterface<typename T>* multiply(T scalar) const
	    {
			MatrixMTL<T, Mx_T>* scaled = new MatrixMTL<T, Mx_T>(this->internal.nrows(),
													this->internal.ncols());
			mtl::copy(this->internal, scaled->internal);
			mtl::scale(scaled->internal, scalar);
			return scaled;
	    }

		/*!
		 *  Scale this by the argument
		 *
		 *  \param scalar The value to scale each element by
		 *
		 */
		void scale(T scalar)
		{
			mtl::scale(internal, scalar);
		}

		/*!
		 *  Retrieve the determinant using LU factorization.
		 *  \todo Check that this is implemented correctly.
		 *
		 *  \return The determinant
		 */
		T determinant() const
		{
			//std::cout << "Huh: " << std::endl;
			//print();
			mtl::dense1D<size_t> pvec(internal.nrows());
			//Mx_T lu(this->internal);
			Mx_T lu(internal.nrows(), internal.ncols());
			mtl::copy(internal, lu);
			mtl::lu_factor<Mx_T, mtl::dense1D<size_t> >(lu, pvec);
			//std::cout << "Now: " << std::endl;
			//print();
			double det = 1.0;
			for (std::size_t i = 0; i < pvec.size(); ++i)
			{
				if (pvec[i] != i)
				{
					det *= -1.0;
					det *= lu(i, i);
				}
			}
			return det;
		}

		//!  Retrieve the native storage type
		Mx_T& native() { return internal; }

		//!  Retrieve the native storage as const reference
		const Mx_T& native() const { return internal; }

		/*!
		 *  Print using MTL print utilities.
		 *
		 */
		void print() const
		{
			mtl::print_all_matrix(internal);
		}
	private:
		Mx_T internal;
	};

	/*!
	 *  \class MatrixCreatorMTL
	 *  \brief Provides implementation for matrix creation powered by MTL
	 *
	 *  Implementation of Matrix creator interface for MTL.  Default template
	 *  arguments allow for the creation of whatever basic type the mtl matrix
	 *  "generator" produces by default, and any other alternatives the user needs (e.g.,
	 *  sparse matrices).
	 */
	template<typename T, typename Mx_T = mtl::matrix<typename T>::type> class MatrixCreatorMTL : public MatrixCreatorPartial<T>
	{
	public:
		// Default constructor
		MatrixCreatorMTL() {}

		// Destructor
		~MatrixCreatorMTL() {}

		/*!
		 *  Create a new MTL matrix of type specified in Mx_T
		 *
		 */
		MatrixInterface<T>* newMatrix(int rows, int cols) const
		{
			return new MatrixMTL<T, Mx_T>(rows, cols);
		}
	};
}
}
}
#endif
