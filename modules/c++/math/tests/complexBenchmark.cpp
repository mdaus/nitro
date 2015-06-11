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
#include <iomanip>
#include <sstream>
#include <complex>
#include <vector>
#include <sys/StopWatch.h>
#include <str/Convert.h>

const size_t NUM_TRIALS = 4;

template<typename T>
void limit(T & num)
{
    while (100 < num)
    {
        num /= 10;
    }
}

std::complex<float> getMeanWComplex(sys::RealTimeStopWatch& wtch,
                                    std::vector<std::complex<float> > in, 
                                    size_t sze, 
                                    double& duration,
                                    size_t numLoops = 1)
{
    std::complex<double> tmp(0.0,0.0);
    wtch.start();
    
    for(size_t j = 0; j < numLoops; ++j)
    {
        for (size_t i = 0; i < sze; ++i)
        {
            tmp += in[i];
        }
    }

    tmp /= sze;
    duration = wtch.stop();
    
    return std::complex<float>(static_cast<float>(tmp.real()),
                               static_cast<float>(tmp.imag()));
}

std::complex<float> getMeanWDouble(sys::RealTimeStopWatch& wtch, 
                                   std::vector<std::complex<float> > in, 
                                   size_t sze, 
                                   double& duration,
                                   size_t numLoops = 1)
{
    double meanI = 0.0;
    double meanQ = 0.0;

    wtch.start();

    for (size_t j = 0; j < numLoops; ++j)
    {
        for (size_t i = 0; i < sze; ++i)
        {
            meanI += in[i].real();
            meanQ += in[i].imag();
        }
    }

    meanI /= (sze * numLoops);
    meanQ /= (sze * numLoops);
    
    duration = wtch.stop();
    std::complex<float> tmp(meanI, meanQ);

    return tmp;

}

void print(std::ostream& out, size_t sze, std::complex<float> meanOne, 
           std::complex<float> meanTwo, double durOne, double durTwo)
{
    out << std::setw(15) << sze
        << std::setw(25) << meanOne 
        << std::setw(25) << meanTwo 
        << std::setw(15) << durOne/1000  
        << std::setw(25) << durTwo/1000 << '\n';
}

void loopingBenchmark(std::ostringstream& out,
                      size_t size,
                      size_t growthFactor,
                      size_t numGrowths)
{
    std::vector<std::complex<float> > arr(size);
    
    srand(time(NULL));

    float real = rand() % 100 + 1;
    float imag = rand() % 100 + 1;

    arr[0] = std::complex<float>(real, imag);
    for (size_t i = 0; i < size; ++i)
    {
        real += arr[i-1].real() + arr[i-1].imag();
        imag += arr[i-1].imag() * arr[i-1].real();

        limit(real);
        limit(imag);

        arr[i] = std::complex<float>(real, imag);
    }

    size_t numLoops = 1;
    for (size_t i = 0; i < numGrowths; ++i)
    {
        sys::RealTimeStopWatch cmplxWatch;
        sys::RealTimeStopWatch dblWatch;

        double cmplxTime;
        double dblTime;

        for(size_t k = 0; k < NUM_TRIALS; ++k)
        {
            std::complex<float> cmplxMean = getMeanWComplex(cmplxWatch,
                                                            arr,
                                                            size,
                                                            cmplxTime,
                                                            numLoops);

            std::complex<float> dblMean = getMeanWDouble(dblWatch,
                                                         arr,
                                                         size,
                                                         dblTime,
                                                         numLoops);

            print(out, size * numLoops, cmplxMean, dblMean, cmplxTime, dblTime);
        }
        numLoops *= growthFactor;

        if (sizeof(std::complex<float>) * size * numLoops > 10E10)
        {
            std::cout << "ending early to prevent growth spiraling" << std::endl;
            return;
        }

    }
}

size_t decideSize(size_t initSize, size_t growthFactor, size_t numGrowths)
{
    //setup size calculation variables
    const size_t MAX_SIZE = 10E10 / (sizeof( std::complex<float>));
    size_t largestPosGrowth = initSize * std::pow(growthFactor, numGrowths);
    size_t largestPosSize = std::min(largestPosGrowth, MAX_SIZE);

    //if growth is too high, find last growth less than MaxSize
    if (largestPosSize == MAX_SIZE)
    {
        //simulate scaling until scaling would exceed MAX_SIZE
        while (largestPosSize * growthFactor < MAX_SIZE)
        {
            largestPosSize *= growthFactor;
        }
    }

    return largestPosSize;
}
void singlePassBenchmark(std::ostringstream& out, 
                         size_t size,
                         size_t growthFactor,
                         size_t numGrowths)
{
    //Determine end size of Vector
    size_t endSize = decideSize(size, growthFactor, numGrowths);
    //Initialize the vector ahead of time and then fill it incrementally
    std::vector<std::complex<float> > arr;
    arr.reserve(endSize);

    srand(time(NULL));

    float real =  rand() % 100 + 1;
    float imag =  rand() % 100 + 1;

    arr.push_back(std::complex<float>(real, imag));
    
    for (size_t j = 1; j < endSize; ++j)
    {
        real += (arr[j-1].real() + arr[j-1].imag());
        imag += (arr[j-1].imag() * arr[j-1].real());
        
        //limit the range of the variables
        limit(real);
        limit(imag);

        arr.push_back(std::complex<float>(real, imag));
    }

    for (size_t i = 0; i < numGrowths; ++i)
    {
    
        sys::RealTimeStopWatch cmplxWatch;
        sys::RealTimeStopWatch dblWatch;

        double cmplxTime;
        double dblTime;
        for (size_t k = 0; k < NUM_TRIALS; ++k)
        {
            std::complex<float> cmplxMean = getMeanWComplex(cmplxWatch,
                                                            arr,
                                                            size,
                                                            cmplxTime);

            std::complex<float> dblMean = getMeanWDouble(dblWatch,
                                                         arr,
                                                         size,
                                                         dblTime);

            print(out, size, cmplxMean, dblMean, cmplxTime, dblTime);
        }
        
        size *= growthFactor;

        if (sizeof(std::complex<float>) * size > 10E10)
        {
            std::cout << "ending early to prevent growth spiraling" << std::endl;
            return;
        }
    }

}
int main(int argc, char** argv)
{
    if (argc != 4 || argc != 5)
    {
        std::cerr << "ERROR, incorrect calling" << std::endl;
        std::cerr << "use ./test <InitialArraySize>  <growthFactor>"
                  << " <numberOfIterations> [Loop?]"
                  << std::endl
                  << "a 1 in the Loop? spot means use looping benchmark"
                  << ".  The looping benchmark changes the behavior and"
                  << " can change the results, but allows for less memory usage"
                  << std::endl;
        return 1;
    }
    size_t sze = atoi(argv[1]);
    size_t growthFactor = atoi(argv[2]);
    size_t numIter = atoi(argv[3]);
    size_t toLoop = 0;
    if (argc == 5)
    {
        toLoop = atoi(argv[4]);
    }
    std::ostringstream out;
    out << std::setprecision(5);
    out.setf(std::ios::fixed);
    out.setf(std::ios::left);

    if (toLoop == 1)
    {
        loopingBenchmark(out, sze, growthFactor, numIter);
    }
    else
    {
        singlePassBenchmark(out, sze, growthFactor, numIter);
    }

    std::cout << out.str() << std::endl;
}
