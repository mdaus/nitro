/* Users guide
    
    To use this test, you must run the executable with 3 extra numbers
    the first number is the size of the initial array
    the second number is the growth factor, each time the size is increased, the old size is multiplied by this number
    the third number is the number of times the size should grow

    example:  ./complexBenchmark 10 2 15
    this will run the test for array sizes 10, 20, 40,... until the array size has been increased 15 times

*/

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <complex>
#include <numeric>
#include <sys/StopWatch.h>

std::complex<float> getMeanWComplex(sys::RealTimeStopWatch& wtch, std::complex<float>* in, long long sze, double& duration)
{
    std::complex<double> tmp(0.0,0.0);
    wtch.start();
    
    for (long long i = 0; i < sze; ++i)
    {
        tmp += in[i];
    }

    tmp /= sze;
    duration = wtch.stop();
    
    return std::complex<float>(static_cast<float>(tmp.real()), static_cast<float>(tmp.imag()));
}

std::complex<float> getMeanWDouble(sys::RealTimeStopWatch& wtch, std::complex<float>* in, long long sze, double& duration)
{
    double meanI = 0.0;
    double meanQ = 0.0;

    wtch.start();
    for (long long i = 0; i < sze; ++i)
    {
        meanI += in[i].real();
        meanQ += in[i].imag();
    }

    meanI /= sze;
    meanQ /= sze;
    
    duration = wtch.stop();
    std::complex<float> tmp(meanI, meanQ);

    return tmp;

}
std::complex<float> getMeanWAcc(sys::RealTimeStopWatch& wtch, std::complex<float>* in, long long sze, double& duration)
{
    std::complex<double> tmp(0.0,0.0);
    wtch.start();
    
    tmp = std::accumulate(in, in + sze, std::complex<float>(0.0,0.0));
    tmp /= sze;

    duration = wtch.stop();
    
    return std::complex<float>(static_cast<float>(tmp.real()), static_cast<float>(tmp.imag()));
}
void print(std::ostream& out, long long sze, std::complex<float> meanOne,
                                             std::complex<float> meanTwo,
                                             std::complex<float> meanThree, 
                                             double durOne, 
                                             double durTwo,
                                             double durThree)
{
    out << std::setw(15) << sze
        << std::setw(25) << meanOne 
        << std::setw(25) << meanTwo 
        << std::setw(25) << meanThree 
        << std::setw(15) << durOne/1000  
        << std::setw(25) << durTwo/1000
        << std::setw(25) << durThree/1000 << '\n';
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "ERROR, incorrect calling" << std::endl;
        std::cerr << "use ./test <InitialArraySize>  <growthFactor> <numberOfIterations>" <<std::endl;
        return 1;
    }
    long long sze = atoi(argv[1]);
    size_t growthFactor = atoi(argv[2]);
    size_t numIter = atoi(argv[3]); 
    std::ofstream outFile("out3.txt");
    std::ostringstream out;
    out << std::setprecision(5);
    out.setf(std::ios::fixed);
    out.setf(std::ios::left);
    
    outFile << std::setprecision(5);
    outFile.setf(std::ios::fixed);
    outFile.setf(std::ios::left);

    //Initialize the array ahead of time and then fill it incrementally
    const unsigned long long MAX_SIZE = 10E10 / (sizeof( std::complex<float>));
    std::complex<float>* arr = new std::complex<float> [MAX_SIZE];
    srand(time(NULL));

    //std::cout << "seeded " << std::endl;

    float Real =  rand() % 100 + 1;
    float Imag =  rand() % 100 + 1;

    //std::cout << "Assigning first spot" << std::endl;
    arr[0] = std::complex<float>(Real, Imag);
    long long j = 1;    
    for (long long i = 0; i < numIter; ++i)
    {

        //std::cout << "creating array" << std::endl;

        //std::cout << "created array" << std::endl;

        for (; j < sze; ++j)
        {
            Real += (arr[j-1].real() + arr[j-1].imag());
            Imag += (arr[j-1].imag() * arr[j-1].real());
            
            //DJS: limit the range of the variables
            while(100 < Real)
            {
                Real /= 10;
            }
            while(100 < Imag)
            {
                Imag /= 10;
            }

            arr[j] = std::complex<float>(Real, Imag);
            //if( i % ( sze/10) == 0)
                //std::cerr << i << std::endl;
        }
        
        //for (size_t i = 0; i < sze; ++i)
        //    std::cout << arr[i] << std::endl;
        
        sys::RealTimeStopWatch cmplxWatch;
        sys::RealTimeStopWatch dblWatch;
        sys::RealTimeStopWatch accWatch;
        double cmplxDuration;
        double dblDuration;
        double accDuration;
        for (size_t k = 0; k < 4; ++k)
        {
            std::complex<float> cmplxMean = getMeanWComplex(cmplxWatch, arr, sze, cmplxDuration);

            std::complex<float> dblMean = getMeanWDouble(dblWatch, arr, sze, dblDuration);

            std::complex<float> accMean = getMeanWAcc(accWatch, arr, sze, accDuration);

            print(out, sze, cmplxMean, dblMean, accMean, cmplxDuration, dblDuration, accDuration);
        }
        //delete [] arr;
        
        sze *= growthFactor;

        if (sizeof(std::complex<float>) * sze > 10E10)
        {
            std::cout << "ending early to prevent large memory usage growth spiraling" << std::endl;
            break;
        }
    }
    delete [] arr;
    outFile << out.str() << std::endl;
    outFile.close();
}
