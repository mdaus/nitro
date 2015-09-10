#include <windows.h>
#include <iostream>
#include <str/Convert.h>

int main(int argc, char **argv)
{
    for (size_t ii = 0; ii < 1000; ++ii)
    {
        std::cout << "Wait [" << str::toString(ii) << "] Finished!" << std::endl;
        Sleep(1000);
    }
    return 0;
}
