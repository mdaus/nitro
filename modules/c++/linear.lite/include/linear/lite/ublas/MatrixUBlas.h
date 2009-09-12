#ifndef __MATRIX_UBLAS_H__
#define __MATRIX_UBLAS_H__

#include "linear/lite/MatrixInterface.h"
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include "linear/lite/ublas/VectorUBlas.h" // mat-vec mult

/*!
 *  \file MatrixUBlas
 *  \brief Operations and classes for uBlas implementation driver
 */

namespace linear
{
	namespace lite
	{
	namespace ublas
	{
	/*!
	 *  \class MatrixUBlas
	 *  \brief Implementation of a MatrixInterface using uBlas (part of boost)
	 *
	 *  This class is a driver for the Matrix<T> which makes use of uBlas underneath.
	 *  uBlas is a templated linear algebra package that has been added to boost.  It supports a variety
	 *  of operations including blas 1, 2 & 3, all implemented in C++.  There are also independent drivers
	 *  available which map fortran libraries such as Atlas and Lapack to uBlas calls.  These implementations
	 *  of uBlas have not been tested against this class.
	 *
	 *  This class takes in the initial MatrixInterface template type, but also adds a storage type default
	 *  typename.  The default will create a dense 2 dimensional matrix.  This allows a user to pass in
	 *  other supported uBlas storage, for example, compressed matrices (although at this time, sparse matrices
	 *  appear to behave incorrectly when compiled into an application using this class).
	 */
	template<typename T, typename Mx_T = boost::numeric::ublas::matrix< typename T > > class MatrixUBlas : public MatrixInterface<T>
	{
	public:
	    //typedef boost::numeric::ublas::matrix< typename T > Mx_T;
	    MatrixUBlas() {}

		/*!
		 *  Construct this object from native matrix type
		 *  \param m A native matrix type
		 */
		MatrixUBlas(const Mx_T& m)
		{
			internal = m;
		}

		/*!
		 *  Construct a matrix with number of rows and columns specified.
		 *  \param rows The number of rows to create
		 *  \param cols The number of columns to create
		 */
	    MatrixUBlas(int rows, int cols)
	    {
			internal = Mx_T(rows, cols);
	    }

		/*!
		 *  Implementation of the clone function for the MatrixUBlas
		 *  derived class
		 *  \return A new base pointer to the derived class
		 */
		MatrixInterface<T>* clone() const
		{
			return new MatrixUBlas<T, Mx_T>(this->internal);
		}

		//!  Destructor
		~MatrixUBlas() {}

		/*!
		 *  Initialize the matrix, resizing rows and columns
		 *  to the new dimensions
		 *  \param rows The number of rows to create
		 *  \param cols The number of cols to create
		 */
		void init(int rows, int cols) { internal.resize(rows, cols); }

		/*!
		 *  Retrieve the number of rows in a matrix
		 *  \return The number of rows
		 */
		int rows() const { return internal.size1(); }

		/*!
		 *  Retrieve the number of cols in a matrix
		 *  \return The number of cols
		 */
	    int cols() const { return internal.size2(); }

		/*!
		 *  Get a copy of the internal element at the ith row and jth column
		 *  \param i The ith row
		 *  \param j The jth column
		 *  \return A copy of the element at the ith row and jth column
		 */
	    T get(int i, int j) const { return internal(i, j); }

		/*!
		 *  Get a reference to the internal element at the ith row and jth column
		 *  \param i The ith row
		 *  \param j The jth column
		 *  \return A reference to the element at the ith row and jth column
		 */
	    //T& get(int i, int j) { return reinterpret_cast<T&>(internal(i, j)); }

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
		 *  Create a transposed matrix of this, dynamically allocated, and
		 *  return it to the caller as a pointer to the base class
		 *  \param i The ith row
		 *  \param j The jth column
		 *  \return A reference to the element at the ith row and jth column
		 */
		MatrixInterface<T>* transpose() const
	    {
			Mx_T transpose = boost::numeric::ublas::trans( this->internal );
			return new MatrixUBlas<T, Mx_T>(transpose);

	    }

		/*!
		 *  Create an inverse matrix of this, dynamically allocated, and
		 *  return it to the caller as a pointer to the base class.  Uninvertible
		 *  matrices may have undefined behavior, most likely, wiil assert.
		 *  \param i The ith row
		 *  \param j The jth column
		 *  \return A reference to the element at the ith row and jth column
		 */
	    MatrixInterface<T>* inverse() const
	    {

			Mx_T a(internal);
			boost::numeric::ublas::permutation_matrix<std::size_t> pmx(a.size1());
			boost::numeric::ublas::lu_factorize(a, pmx);
			Mx_T inv(boost::numeric::ublas::identity_matrix<double>(a.size1()));
			//inv.assign(boost::numeric::ublas::identity_matrix<double>(a.size1()));
			boost::numeric::ublas::lu_substitute(a, pmx, inv);
			return new MatrixUBlas<T, Mx_T>(inv);
	    }

		/*!
		 *  Matrix addition.  Produces a new matrix which is the sum of this  and
		 *  the argument parameter.  This method will cast the matrix to this derived
		 *  class before performing addition.
		 *
		 *  \param mx The matrix to add to this
		 *  \return A new allocated matrix which is the sum of this and the argument
		 */
		MatrixInterface<T>* add(const MatrixInterface<T>& mx) const
		{
			const MatrixUBlas<T, Mx_T>& reint = down_cast<const MatrixUBlas<T, Mx_T>&, const MatrixInterface<T>& >(mx);
			return new MatrixUBlas<T, Mx_T>(internal + reint.internal);
		}

		/*!
		 *  Post-multiply this and mx, producing a new allocated matrix which
		 *  is returned to the caller.  This method will cast the matrix to this derived
		 *  class before performing multiplication.
		 *
		 *  \param mx Matrix to multiply this by
		 *  \return A new-allocated product matrix
		 */
	    MatrixInterface<T>* multiply(const MatrixInterface<T>& mx) const
	    {

			const MatrixUBlas<T, Mx_T>& reint = down_cast< const MatrixUBlas<T, Mx_T>&, const MatrixInterface<T>& >(mx);
			Mx_T p = boost::numeric::ublas::prod(this->internal, reint.internal);
			return new MatrixUBlas<T, Mx_T>(p);
	    }

		/*!
		 *  Multiply this matrix by a vector, producing a new allocated vector which
		 *  is returned to the caller.  This method will cast the argument to the
		 *  derived uBlas vector implementation class before performing multiplication.
		 *
		 *  \param vec Matrix to multiply this by
		 *  \return A new-allocated product matrix
		 */
		VectorInterface<T>* multiply(const VectorInterface<T>* vec) const
		{
			const VectorUBlas<T>* reint = down_cast<const VectorUBlas<T>*, const VectorInterface<T>* >(vec);
			assert(reint); // pointer can be checked!
			assert(cols() == reint->size());
			VectorUBlas<T>::Arr_T p = boost::numeric::ublas::prod(this->internal, reint->native());
			return new VectorUBlas<T>(p);
		}
		/*!
		 *  Multiply this matrix by a scalar value, producing a new allocated matrix which
		 *  is returned to the caller.
		 *
		 *  \param scalar value to scale by
		 *  \return A new-allocated product matrix
		 */
	    MatrixInterface<T>* multiply(T scalar) const
	    {
			Mx_T prod = this->internal * scalar;
			return new MatrixUBlas<T, Mx_T>(prod);
	    }
		/*!
		 *  Scale this by a scalar value.
		 *
		 *  \param scalar Matrix to multiply this by
		 */
		void scale(T scalar)
	    {
			this->internal *= scalar;
	    }
		/*!
		 *  Find and retrieve the determinant.  Uses LU-factorization.
		 *  \return The determinant
		 */
		T determinant() const
	    {
			Mx_T lu(internal);
			boost::numeric::ublas::permutation_matrix<std::size_t> pvec(lu.size1());
			boost::numeric::ublas::lu_factorize(lu, pvec);
			double det = 1.0;
			for (std::size_t i = 0; i < pvec.size(); ++i)
			{
				if (pvec(i) != i)
				{
					det *= -1.0;
					det *= lu(i, i);
				}
			}
			return det;
	    }
		//!  Retrieve the ublas native storage (non-const)
		Mx_T& native() { return internal; }
		//!  Retrieve the ublas native storage (const)
		const Mx_T& native() const { return internal; }

		/*!
		 *  Print the matrix represented in ublas
		 */
		void print() const
		{
			std::cout << internal << std::endl;
		}
	private:
	    Mx_T internal;
	};

	/*!
	 *  \class MatrixCreatorUBlas
	 *  \brief Creation mechanism for uBlas-backed matrices
	 *
	 *  Produces uBlas matrices.  The current implementations for constant value and identity matrix initialization
	 *  should be more efficient than the partial implementation, which is why this class directs the
	 *  MatrixCreatorInterface directly instead, however, for sparse matrices, this seems to cause the creation of
	 *  a temporary, very large, dense matrix to assign from, which is highly undesireable.  For that reason, all
	 *  of the methods in this class are left as virtual (to allow efficient subclassing for sparse matrices)
	 */
	template<typename T, typename Mx_T = boost::numeric::ublas::matrix< typename T > > class MatrixCreatorUBlas :
		public MatrixCreatorInterface<T>
	{
	public:
		MatrixCreatorUBlas() {}
		virtual ~MatrixCreatorUBlas() {}

		/*!
		 *  Create a new matrix with number of rows and columns specified.
		 *
		 *  \return A dynamically allocated Matrix of type T
		 */
		virtual MatrixInterface<T>* newMatrix(int rows, int cols) const { return new MatrixUBlas<T, Mx_T>(rows, cols); }

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
		virtual MatrixInterface<T>* newConstantMatrix(int rows, int cols, T constantValue) const
		{
			return new MatrixUBlas<T, Mx_T>(boost::numeric::ublas::scalar_matrix<T>(rows, cols, constantValue));
		}

		/*!
		 *  Create a new square identity matrix with cols and rows specified as
		 *  single argument.
		 *
		 *  \param d The number of rows and cols of this square matrix.
		 *  \return A dynamically allocated Matrix of type T
		 */
		virtual MatrixInterface<T>* newIdentityMatrix(int d) const
		{
			return new MatrixUBlas<T, Mx_T>(boost::numeric::ublas::identity_matrix<T>(d));
		}

	};
}
}
}
#endif
