#include "linear/lite/LinearAlgebraDriver.h"
#include "linear/lite/Vector.h"
namespace linmath = linear::lite;

int main()
{
	linmath::Vector<double> v1 = linmath::VectorCreator().newConstantVector(3, 0);
	linmath::Vector<double> v2 = linmath::VectorCreator().newConstantVector(3, 0);
	
	v1[0] = 1.0;
	v2[1] = 1.0;

	linmath::Vector<double> v3 = v1.cross(v2);
	v3.print();
	
	v3.scale(0.4);
	v3.print();


	linmath::Vector<double> v4 = v3.add(v1);
	v4.print();

	linmath::Vector<double> v5 = v1 + v2;
	v5.print();
	
	std::cout << "L2 Norm: " << v4.norm2() << std::endl;
	std::cout << "v3 dot v4: " << v3.dot(v4) << std::endl;

	linmath::Vector<double> v6 = v1 * -1.0;
	
	linmath::Vector<double> v7 = -1.0 * v1;
	v6.print();
	v7.print();
	
	v1 *= -1.0;
	std::cout << v1 << std::endl;
	v1 /= -1.0;
	std::cout << v1 << std::endl;
        linmath::Vector<double> _1d = 
            linmath::VectorCreator().newConstantVector(1, -2.0);
	std::cout << _1d << std::endl;

        linmath::Vector<double> v8 = v7.clone();
        v8.print();

        double* vals = new double[3];
        v8.get(vals);

        for(int i=0; i<3; i++)
        {
           cout << vals[i] << " ";
        }
        cout << endl;
        delete vals;

        vals = new double[5];
        for(int i=0; i<5; i++)
        {
           vals[i] = 5;
        }
        v8.set(5, vals);
        v8.print();
        v7.print();


        linmath::Vector<double>* v9 = NULL; 
        std::auto_ptr<linmath::Vector<double> > vAuto;

        linmath::Vector<double>tmp = linmath::VectorCreator().newVector(1);

        v9 = &tmp;
        vAuto.reset(&tmp);

        v9->set(5, vals);
        v9->print();
        vAuto->print();
        delete vals;

        v9->normalize();
        v9->print();

        vAuto.release();

	return 0;
	
}
