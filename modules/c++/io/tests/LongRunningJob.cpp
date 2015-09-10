#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <iostream>
#include <str/Convert.h>

int main(int argc, char **argv)
{
    for (size_t ii = 0; ii < 1000; ++ii)
    {
        std::cout << "Wait [" << str::toString(ii) << "] Finished!" << std::endl;
#ifdef WIN32
        Sleep(1000);
#else
        sleep(1);
#endif
    }
    return 0;
}
