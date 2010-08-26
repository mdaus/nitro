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
#include "linear/lite/LinearAlgebraDriver.h"
#include "linear/lite/Matrix.h"

using namespace linear::lite;

void show(const std::string& name)
{
    std::cout << std::endl << "======================= " << name
            << " =======================" << std::endl;
}
void initVec(VectorInterface<double>& v)
{
    v[0] = 0.3;
    v[1] = 0.6;
    v[2] = 0.9;
    v.print();
    std::cout << v.size() << std::endl;
}

void jumpThroughHoops(VectorInterface<double>& v)
{
    v.print();

    double dp = v.dot(v);
    std::cout << dp << std::endl;
    std::cout << v.norm2() << std::endl;

}

void initMx(linear::lite::Matrix<double>& m)
{
    m.scale(0.62);
}

void jumpThroughHoops(const linear::lite::Matrix<double>& m)
{
    std::cout << "Start: " << std::endl;
    m.print();
    //math::MatrixInterface<double>* mx = mtl.inverse();

    double det = m.determinant();
    std::cout << "Det: " << det << std::endl;
    //if (det)
    //	m.inverse();

    m.print();

    std::cout << "Tr: " << m.trace() << std::endl;

    linear::lite::Matrix<double> trans = m.transpose();

    linear::lite::Matrix<double> x = trans;

    std::cout << "Trans: " << std::endl;
    trans.print();
    x(0, 1) = 4;

    linear::lite::Matrix<double> prod = trans.multiply(x);
    //x.print();
    std::cout << "Prod: " << std::endl;
    prod.print();

    linear::lite::Matrix<double> altP = trans * x;
    std::cout << altP << std::endl;

    linear::lite::Matrix<double> sum = prod.add(m);
    std::cout << "Sum: " << std::endl;
    sum.print();
    linear::lite::Matrix<double> altS = prod + m;
    std::cout << altS << std::endl;
    linear::lite::Vector<double> vec = VectorCreator().newConstantVector(4, 1);
    std::cout << altS * vec << std::endl;
    std::cout << vec << std::endl;

    linear::lite::Matrix<double> vecx = MatrixCreator().newConstantMatrix(4, 1,
                                                                          1);
    std::cout << vecx << std::endl;
    std::cout << altS * vecx << std::endl;
}
int main()
{
    show( LINEAR_LITE_DRIVER);
    linear::lite::Matrix<double> myMx = MatrixCreator().newConstantMatrix(4, 4,
                                                                          1);
    initMx(myMx);
    jumpThroughHoops(myMx);

#ifdef HAVE_SPARSE
    show("sparse matrix");

    Matrix<double> sparse = SparseMatrixCreator().newMatrix(10, 10);
    for (int i = 0; i < 10; i++)
    for(int j = 0; j < 10; j++)
    sparse.set(i, j, 0);// = 0;

    for (int i = 0; i < 10; i++)
    sparse.set(i, i, 1);// = 1;

    sparse.scale(14.2);
    sparse.print();
#endif

    linear::lite::Vector<double> v1 = VectorCreator().newConstantVector(3, 0);
    linear::lite::Vector<double> v2 = VectorCreator().newConstantVector(3, 0);

    v1[0] = 1;
    v2[1] = 1;
    v1.print();
    v2.print();

    linear::lite::Vector<double> v3 = v1.cross(v2);
    v3.print();

    v3.scale(0.4);
    v3.print();

    linear::lite::Vector<double> v4 = v3.add(v1);
    v4.print();
    std::cout << "L2 Norm: " << v4.norm2() << std::endl;
    std::cout << "v3 dot v4" << v3.dot(v4) << std::endl;

    try
    {
        myMx.print();
        double* vals = new double[4];
        vals[0] = 1.0;
        vals[1] = 2.0;
        vals[2] = 3.0;
        vals[3] = 4.0;
        linear::lite::Vector<double> v5 = VectorCreator().newVector(4, vals);
        delete vals;
        myMx.appendColumnVector(v5);
        myMx.print();

        linear::lite::Matrix<double> myMx2 = myMx.clone();

        myMx2.appendColumnVector(v5);
        myMx.appendColumns(myMx2);
        myMx.print();

        linear::lite::Matrix<double> myMx3 = MatrixCreator().newMatrix(0, 0);
        myMx.getColumns(3, 5, myMx3);
        myMx3.print();

        linear::lite::Matrix<double> myMx4 = MatrixCreator().newMatrix(0, 0);
        myMx.getColumns(0, 10, myMx4);
        myMx4.print();

        myMx.print();
        linear::lite::Vector<double> v6 = VectorCreator().newVector(4);
        myMx.getColumnVector(0, v5);
        myMx.getRowVector(0, v6);
        v5.print();
        v6.print();

        v5[2] = 7;
        v6[3] = 7;
        myMx.setColumnVector(6, v5);
        myMx.print();
        myMx.setRowVector(2, v6);
        myMx.print();

        double* outVals = new double[4 * 11];
        myMx.get(outVals);

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 11; j++)
            {
                cout << outVals[i * 11 + j] << " ";
            }
            cout << endl;
        }

        outVals[5] = 7;
        outVals[6] = 7;
        outVals[7] = 7;
        outVals[8] = 7;
        outVals[16] = 7;
        outVals[17] = 7;
        outVals[18] = 7;
        outVals[19] = 7;
        outVals[27] = 7;
        outVals[28] = 7;
        outVals[29] = 7;
        outVals[30] = 7;

        myMx.set(4, 11, outVals);
        myMx.print();

        delete outVals;

        outVals = new double[4 * 4];
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                outVals[i * 4 + j] = 1;
            }
        }

        myMx.set(4, 4, outVals);
        myMx.print();
        myMx2.print();

        linear::lite::Matrix<double>* myMx5;
        std::auto_ptr < linear::lite::Matrix<double>> mAuto;

        linear::lite::Matrix<double> tmp = MatrixCreator().newMatrix(0, 0);

        myMx5 = &tmp;
        mAuto.reset(&tmp);

        myMx5->print();
        myMx5->set(4, 4, outVals);
        myMx5->print();
        delete outVals;

        vals = new double[4];
        vals[0] = 1.0;
        vals[1] = 2.0;
        vals[2] = 3.0;
        vals[3] = 4.0;
        v5.set(4, vals);
        delete vals;
        v5.print();
        myMx5->appendRowVector(v5);
        myMx5->print();

        linear::lite::Matrix<double> myMx6 = myMx5->clone();
        myMx5->appendRows(myMx6);
        myMx5->print();
        mAuto->print();
        myMx6.print();

        mAuto.release();
    }
    catch (except::Exception &e)
    {
        cout << e.toString() << endl;
    }

    return 0;
}
