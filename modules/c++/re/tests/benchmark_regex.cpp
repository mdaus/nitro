
#include <string>
#include <fstream>

#include <import/re.h>
#include <sys/StopWatch.h>
#include <sys/LocalDateTime.h>

// First just benchmark the Regex creation time (including compile)
double BM_RegexCreation(uint64_t numIterations)
{
    sys::RealTimeStopWatch sw;

    sw.start();
    for (uint64_t ii = 0; ii < numIterations; ++ii)
    {
        re::Regex regex("beam(Id|String)");
    }
    double elapsedTimeMS = sw.stop();

    return elapsedTimeMS / numIterations;
}


// Now benchmark the actual string-matching
double BM_RegexMatch(uint64_t numIterations, const std::string& fileString)
{
    sys::RealTimeStopWatch sw;

    // should match multiple times
    re::Regex regex("beam(Id|String)");

    // should never match
    //re::Regex regex("foobarfighters");

    sw.start();
    for (uint64_t ii = 0; ii < numIterations; ++ii)
    {
        regex.matches( fileString );
    }
    double elapsedTimeMS = sw.stop();

    return elapsedTimeMS / numIterations;
}


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./bm_regex inputFile numIterations" << std::endl;
        return 1;
    }

    uint64_t numIterations = std::stoi(argv[2]);

    sys::LocalDateTime ldt;
    std::cout << ldt.format(std::string("%Y-%m-%d %H:%M:%S")) << std::endl;
    
    // Open our text file and feed it into the static buffer
    std::ifstream bigFin(argv[1]);
    if (!bigFin.is_open())
    {
        std::cerr << "Error opening text file!" << std::endl;
        return 2;
    }
    
    size_t size = bigFin.tellg();
    std::string fileString(size, '\0');
    
    bigFin.seekg(0);
    bigFin.read(&fileString[0], size);
    bigFin.close();

    // Now run the benchmarks
    double swtime0 = BM_RegexCreation(numIterations);
    double swtime1 = BM_RegexMatch(numIterations, fileString);

    // Convert ms to ns
    swtime0 *= 1.e6;
    swtime1 *= 1.e6;

    // Pretty-print our results
    char outbuff[10000];
    sprintf(outbuff, "%-20s %20s %15s\n", "Benchmark", "Time/Iteration (ns)", "Iterations");
    std::cout << outbuff;
    std::string line(57, '-');
    std::cout << line << std::endl;
    
    sprintf(outbuff, "%-20s %20.0lf %15ld\n", "BM_RegexCreation", swtime0, numIterations);
    std::cout << outbuff;
    sprintf(outbuff, "%-20s %20.0lf %15ld\n", "BM_RegexMatch", swtime1, numIterations);
    std::cout << outbuff;

    return 0;
}
