/* =========================================================================
 * This file is part of math-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 *
 * math.linear-c++ is free software; you can redistribute it and/or modify
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

/* Users guide

    To use this test, you must run the executable with 3 extra numbers
    the first number is the size of the initial array
    the second number is the growth factor, each time the size is increased,
        the old size is multiplied by this number
    the third number is the number of times the size should grow
    the fourth number (optional) denotes whether to loop or not, a 1 indicates
        to loop

    examples:
    ./complexBenchmark 10 2 15
        --this will run the test for array sizes 10, 20, 40,... until the array
            size has been increased 15 times

    ./complexBenchmark 10 2 15 1
        --runs the same as above, but conserves memory by looping to simulate
            increasing the size of the array

*/

/*  Results:
        When using the non looping benchmark, aka for large continuous data 
    sets, it was faster to use doubles.  However when repeatedly looping over
    a set of data, it is faster use complex numbers.  

        When using the non looping method, the usage of doubles was
    approximately .5% faster than the usage of complex numbers.  

        When using the looping method, the usage of complex numbers was faster
    by approximatley 5% than the usage of doubles.

    This comparison is only valid when on linux though using g++ 4.9.  Tests
    were not runnable at sufficiently large scales on windows due to memory
    constraints.

        In summmary, it is best to use complex numbers for iteration over an
    array.  The speed gains from using doubles is minimal at best, and the usage
    of complex numbers both increases clarity, code readability, and does not
    incur a performance cost.

*/

#include <algorithm>
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

//Limits num so that it is less than 100
template<typename T>
void limit(T & num)
{
    while (100 < num)
    {
        num /= 10;
    }
}


/*
 *  \purpose
 *      Determine the mean of the vector in, using complex addition
 *      Time how long the operation takes
 *
 *  \params
 *      wtch: a sys::RealTimeStopWatch that performs the timing operations
 *      in: the vector whose mean is being found
 *      sze: the size of the vector
 *      numLoops: allows for use with looping benchmark, specifies loop amount
 *
 *  \output
 *      duration: The amount of time the operation took according to wtch
 *      return value: The mean of the vector
 */
std::complex<float> getMeanWComplex(sys::RealTimeStopWatch& wtch,
                                    const std::vector<std::complex<float> >& in,
                                    size_t sze,
                                    double& duration,
                                    size_t numLoops = 1)
{
    //declare starting point
    std::complex<double> tmp(0.0,0.0);

    //start the watch
    wtch.start();

    //find the mean
    for(size_t j = 0; j < numLoops; ++j)
    {
        for (size_t i = 0; i < sze; ++i)
        {
            tmp += in[i];
        }
    }

    tmp /= static_cast<double>(sze * numLoops);

    //stop the watch and record the duration
    duration = wtch.stop();

    //return the mean
    return std::complex<float>(static_cast<float>(tmp.real()),
                               static_cast<float>(tmp.imag()));
}

/*
 *  \purpose
 *      Determine the mean of the vector in, using addition of real
 *          and imaginary parts separately
 *      Time how long the operation takes
 *
 *  \params
 *      wtch: a sys::RealTimeStopWatch that performs the timing operations
 *      in: the vector whose mean is being found
 *      sze: the size of the vector
 *      numLoops: allows for use with looping benchmark, specifies loop amount
 *
 *  \output
 *      duration: The amount of time the operation took according to wtch
 *      return value: The mean of the vector
 */
std::complex<float> getMeanWDouble(sys::RealTimeStopWatch& wtch,
                                   const std::vector<std::complex<float> >& in,
                                   size_t sze,
                                   double& duration,
                                   size_t numLoops = 1)
{
    //declare starting values
    double meanI = 0.0;
    double meanQ = 0.0;

    //start the watch
    wtch.start();

    //find the mean
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

    //stop the watch and record the duration;
    duration = wtch.stop();

    //return the mean
    return std::complex<float>(static_cast<float>(meanI), static_cast<float>(meanQ));

}

//Prints out the results in a table format
void print(std::ostream& out, size_t sze, std::complex<float> meanOne,
           std::complex<float> meanTwo, double durOne, double durTwo)
{
    out << std::setw(15) << sze
        << std::setw(25) << meanOne
        << std::setw(25) << meanTwo
        << std::setw(15) << durOne/1000
        << std::setw(25) << durTwo/1000 << '\n';
}


/*
 *  \purpose
 *      Run the looping versioin of the benchmark
 *
 *  \params
 *      size: the size of the vector
 *      growthFactor: factor that the vector's size is simulated to grow by
 *          with each iteration
 *      numGrowths: The number of growths before ending the simulation
 *
 *  \output
 *      out: an ostringstream that stores the output from the simulations
 */
void loopingBenchmark(size_t size,
                      size_t growthFactor,
                      size_t numGrowths,
                      std::ostringstream& out)
{
    //declare the vector
    std::vector<std::complex<float> > arr(size);

    //fill the vector based on a random number
    srand(static_cast<unsigned int>(time(NULL)));

    auto real = static_cast<float>(rand() % 100 + 1);
    auto imag = static_cast<float>(rand() % 100 + 1);

    arr[0] = std::complex<float>(real, imag);
    for (size_t i = 0; i < size; ++i)
    {
        real += arr[i-1].real() + arr[i-1].imag();
        imag += arr[i-1].imag() * arr[i-1].real();

        limit(real);
        limit(imag);

        arr[i] = std::complex<float>(real, imag);
    }

    //run the simulation
    size_t numLoops = 1;
    for (size_t i = 0; i < numGrowths; ++i)
    {
        sys::RealTimeStopWatch cmplxWatch;
        sys::RealTimeStopWatch dblWatch;

        double cmplxTime;
        double dblTime;

        for(size_t k = 0; k < NUM_TRIALS; ++k)
        {
            //find the complex mean
            std::complex<float> cmplxMean = getMeanWComplex(cmplxWatch,
                                                            arr,
                                                            size,
                                                            cmplxTime,
                                                            numLoops);
            //find the mean using doubles
            std::complex<float> dblMean = getMeanWDouble(dblWatch,
                                                         arr,
                                                         size,
                                                         dblTime,
                                                         numLoops);
            //output the results
            print(out, size * numLoops, cmplxMean, dblMean, cmplxTime, dblTime);
        }
        //simulate vector size growth
        numLoops *= growthFactor;

        //return if growth simulated would be too large to handle
        if (sizeof(std::complex<float>) * size * numLoops > 10E10)
        {
            std::cout << "ending early to prevent growth spiraling" << std::endl;
            return;
        }

    }
}

//determines how large the vector will actually grow for preallocation
size_t decideSize(size_t initSize, size_t growthFactor, size_t numGrowths)
{
    //setup size calculation variables
    const auto MAX_SIZE = static_cast<size_t>(10E10 / (sizeof( std::complex<float>)));
    auto largestPosGrowth = static_cast<size_t>(
        initSize * std::pow(static_cast<double>(growthFactor),
                            static_cast<double>(numGrowths)));
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

/*
 *  \purpose
 *      Run the singlePass version of the benchmark
 *
 *  \params
 *      size: the size of the vector
 *      growthFactor: factor that the vector's size grows by with each iteration
 *      numGrowths: The number of growths before ending the simulation
 *
 *  \output
 *      out: an ostringstream that stores the output from the simulations
 */
void singlePassBenchmark(size_t size,
                         size_t growthFactor,
                         size_t numGrowths,
                         std::ostringstream& out)
{
    //Determine end size of Vector
    size_t endSize = decideSize(size, growthFactor, numGrowths);

    //Initialize the vector ahead of time and then fill it
    std::vector<std::complex<float> > arr;
    arr.reserve(endSize);

    srand(static_cast<unsigned int>(time(NULL)));

    auto real = static_cast<float>(rand() % 100 + 1);
    auto imag = static_cast<float>(rand() % 100 + 1);

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

    //run the simulation
    for (size_t i = 0; i < numGrowths; ++i)
    {

        sys::RealTimeStopWatch cmplxWatch;
        sys::RealTimeStopWatch dblWatch;

        double cmplxTime;
        double dblTime;
        for (size_t k = 0; k < NUM_TRIALS; ++k)
        {
            //find the mena using complex values
            std::complex<float> cmplxMean = getMeanWComplex(cmplxWatch,
                                                            arr,
                                                            size,
                                                            cmplxTime);
            //find the mean using doubles
            std::complex<float> dblMean = getMeanWDouble(dblWatch,
                                                         arr,
                                                         size,
                                                         dblTime);
            //output the results
            print(out, size, cmplxMean, dblMean, cmplxTime, dblTime);
        }

        //increase size of vector
        size *= growthFactor;

        //return if growth gets too large
        if (sizeof(std::complex<float>) * size > 10E10)
        {
            std::cout << "ending early to prevent growth spiraling" << std::endl;
            return;
        }
    }

}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5)
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

    //read in commandline arguements
    size_t sze = str::toType<size_t>(argv[1]);
    size_t growthFactor = str::toType<size_t>(argv[2]);
    size_t numIter = str::toType<size_t>(argv[3]);
    size_t toLoop = 0;
    if (argc == 5)
    {
        toLoop = str::toType<size_t>(argv[4]);
    }

    //setup ostringstream
    std::ostringstream out;
    out << std::setprecision(5);
    out.setf(std::ios::fixed);
    out.setf(std::ios::left);

    //run the appropriate simulation
    if (toLoop == 1)
    {
        loopingBenchmark(sze, growthFactor, numIter, out);
    }
    else
    {
        singlePassBenchmark(sze, growthFactor, numIter, out);
    }

    //output the results
    std::cout << out.str() << std::endl;
}
