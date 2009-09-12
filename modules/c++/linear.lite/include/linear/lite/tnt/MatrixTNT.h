#ifndef __MATRIX_TNT_H__
#define __MATRIX_TNT_H__

#include <iostream>

#include "tnt.h"
#include "tnt_array2d.h"
#include "jama_svd.h"
//#include "jama_eig.h"
#include "jama_lu.h"
#include "linear/lite/MatrixInterface.h"
#include "linear/lite/tnt/VectorTNT.h" /* used in mat-vec mult */

/*!
 *  \file MatrixTNT.h
 *  \brief TNT/JAMA driver for matrices
 *
 */

namespace linear
{
	namespace lite
	{
	namespace tnt
	{
	/*!
	 *  \class MatrixTNT
	 *  \brief An implementation of a MatrixInterface using the Templated Numerical Toolkit (TNT)
	 *
	 *  This class provides an implementation for matrices in linalg.lite using TNT and JAMA.
	 *  TNT is in a transitionary state, between its old "deprecated" API, which included TNT::Vector
	 *  and TNT::Matrix objects, and the new API using TNT::Array2D objects.  Unfortunately, some
	 *  of the methods are now redundant in this driver, to prevent us from using deprecated code.
	 *  It is anticipated that future version of TNT may offer basic operations for arrays which
	 *  are currently supported in the deprecated APIs.
	 *
	 *
	 */
	template<typename T> class MatrixTNT : public MatrixInterface<T>
	{
	public:
	    typedef TNT::Array2D<T> Mx_T;
		//! Default constructor
		MatrixTNT() {}

		/*!
		 *  Constructor from an existing TNT matrix
		 *
		 *  \param m A TNT data structure to initialize from
		 */
		MatrixTNT(const Mx_T& m)
		{
			internal = m.copy();
		}

		/*!
		 *  Constructor to initialize a matrix with rows
		 *  and columns given.
		 *
         *  \param rows The number of rows
		 *  \param cols The number of columns
		 *
		 */
	    MatrixTNT(int rows, int cols)
	    {
			internal = Mx_T(rows, cols);
	    }

		//!  Destructor
		~MatrixTNT() {}

		/*!
		 *  Deep copy this to a new matrix.  Return via a pointer
		 *  to the base class.
		 *  \return A deep copy of this
		 */
		MatrixInterface<T>* clone() const
		{
			MatrixTNT<T> *mtl = new MatrixTNT(this->internal);
			return mtl;
		}

		/*!
		 *  Initialize the matrix with number of rows and columns specified.
		 *  \param rows The number of rows
		 *  \param cols The number of columns
		 */
		void init(int rows, int cols) { internal = Mx_T(rows, cols); }

		/*!
		 *  Retrieves the number of rows.
		 *
		 */
	    int rows() const { return internal.dim1(); }
		/*!
		 *  Retrieves the number of rows.
		 *
		 */
	    int cols() const { return internal.dim2(); }
		/*!
		 *  Retrieves a copy of the element at specified row-column
		 *  position.
		 *  \param i The row position
		 *  \param j The column position
		 *  \return A copy of the element at the row and column position
		 */
	    T get(int i, int j) const { return internal[i][j]; }
		/*!
		 *  Retrieves a reference to the element at specified row-column
		 *  position.
		 *
		 *  \sa MatrixInterface::set
		 *  \param i The row position
		 *  \param j The column position
		 *  \return A reference to the element at the row and column position
		 */
	    T& get(int i, int j) { return internal[i][j]; }
		void set(int i, int j, const T& elem) { internal[i][j] = elem; }
		/*!
		 *  Return a new MatrixTNT which is the transpose of this.
		 *  TNT used to support this in their matrices, but the Array
		 *  API doesnt appear to, so we do it the long way...
		 *
		 *  \return A new matrix transpose
		 */
		MatrixInterface<T>* transpose() const
		{

			Mx_T trans(internal.dim2(), internal.dim1());
			for (int i = 0; i < internal.dim1(); i++)
				for (int j = 0; j < internal.dim2(); j++)
					trans[j][i] = internal[i][j];
			return new MatrixTNT(trans);
		}

		/*!
		 *  Return a new MatrixTNT which is the sum of this and its argument
		 *  \return A new matrix sum
		 */
		MatrixInterface<T>* add(const MatrixInterface<T>& mx) const
	    {
			const MatrixTNT<T>& reint = down_cast<const MatrixTNT<T>&, const MatrixInterface<T>& >(mx);
			MatrixTNT<T> *prod = new MatrixTNT<T>(internal);
			prod->internal += reint.internal;
			return prod;
	    }
		/*!
		 *  Return a new MatrixTNT which is the product of this and its argument.
		 *
		 *  \return A new matrix product
		 */
		MatrixInterface<T>* multiply(const MatrixInterface<T>& mx) const
	    {
			const MatrixTNT<T>& reint = down_cast<const MatrixTNT<T>&, const MatrixInterface<T>& >(mx);
			MatrixTNT<T> *prod = new MatrixTNT<T>();
			prod->internal = TNT::matmult(this->internal, reint.internal);
			return prod;
	    }

		/*!
		 *  Return a new VectorTNT which is the sum of this and its argument.
		 *  TNT used to support this operation for Matrix and Vector objects,
		 *  but they have deprecated that API, so we do it the long way...
		 *
		 *  \return A new vector product
		 */
		VectorInterface<T>* multiply(const VectorInterface<T>* vec) const
		{

			assert(cols() == vec->size());
			int m = rows();
			int n = cols();

			VectorInterface<T>* prod = new VectorTNT<T>(m);
			T sum;

			for (int i = 0; i < m; i++)
			{
				sum = 0;
				for (int j = 0; j < n; j++)
					sum = sum +  internal[i][j] * (*vec)[j];

				prod->set(i,  sum);
			}

			return prod;
		}
		/*!
		 *  Return a new matrix which is a scaled version of this
		 *  \param scalar The scale value to apply
		 *  \return Return a new Matrix
		 */
	    MatrixInterface<T>* multiply(T scalar) const
	    {
			TNT::Array2D<T> scale(internal.dim1(), internal.dim2(), T(scalar));
			MatrixTNT<T> *prod = new MatrixTNT<T>();
			prod->internal = this->internal * scale;
			return prod;
	    }

		/*!
		 *  Scale this by the amount specified.  We do this the long way
		 *  to prevent copying.
		 *
		 *  \param scalar The scale value to apply
		 */
		void scale(T scalar)
		{
			// implemented slightly different than above to prevent lots of mx creation/copies
			for (int i = 0; i < internal.dim1(); i++)
				for (int j = 0; j < internal.dim2(); j++)
					internal[i][j] *= scalar;
		}

		/*!
		 *  Return a new matrix which is the inverse of this.  Uses LU-factorization
		 *  internally.  If no matrix exists, assertion failure occurs.
		 *  \todo Consider throwing an exception if matrix is not invertible
		 *
		 *  \return A new matrix inverse
		 */
		MatrixInterface<T>* inverse() const
	    {
			MatrixTNT<T> *inv = new MatrixTNT<T>(rows(), cols());


                        TNT::Array2D<T> a(internal.dim1(), internal.dim2(), (T)0);
			for (int i = 0; i < internal.dim1(); i++) a[i][i] = 1;
			JAMA::LU<T> lu(internal);

			if (! lu.det() )
                            throw except::Exception(Ctxt("Matrix not invertible"));

			inv->internal = lu.solve(a);
			return inv;
		}

	   /*!
		*  Finds the determinant using LU-factorization.
		*  \return The determinant of this
		*/
		T determinant() const
		{
			JAMA::LU<T> lu(internal);
			return lu.det();
		}
		//!  Retrieve the native TNT type (non-const)
		Mx_T& native() { return internal; }
		//!  Retrieve the native TNT type (const)
		const Mx_T& native() const { return internal; }

		/*!
		 *  Print using TNT Array representation.
		 *
		 */
		void print() const
		{
			std::cout << internal << std::endl;
		}
	private:

	    Mx_T internal;
	};

	/*!
	 *  \class MatrixCreatorTNT
	 *  \brief Creator pattern for TNT matrix implementations
	 *
	 *  This classes produces dense 2D TNT-backed matrices of type T
	 */
	template<typename T> class MatrixCreatorTNT : public MatrixCreatorPartial<T>
	{
	public:
		//! Default constructor
		MatrixCreatorTNT() {}
		//! Destructor
		~MatrixCreatorTNT() {}

	   /*!
		*  Produce a new matrix of type, and number of rows and cols
		*  specified
		*
		*  \param rows The number of rows
		*  \param cols The number of columns
		*  \return A new TNT-backed matrix
		*/
		MatrixInterface<T>* newMatrix(int rows, int cols) const { return new MatrixTNT<T>(rows, cols); }

                MatrixInterface<T>* newMatrix(int rows, int cols, T* dataPtr) const
                {
                   return MatrixCreatorPartial<T>::newMatrix(rows, cols, dataPtr);
                }

	};
}
}
}
#endif

