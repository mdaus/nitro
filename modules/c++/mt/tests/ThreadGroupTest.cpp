#include <iostream>
#include <cmath>
#include <import/sys.h>
#include <import/mt.h>
using namespace sys;
using namespace mt;
using namespace std;

const int NUM_TASKS = 4;

class MyRunTask : public Runnable
{
public:
    double result;
    
    MyRunTask()
    {
        result = 1;
    }
    virtual ~MyRunTask()
    {
    }

    virtual void run()
    {
		for (int count = 0; count < 10000000; count++)
		{
			result = sin((double) count);
		}
    }
};

void print(MyRunTask *tasks[NUM_TASKS])
{
    for (int i = 0; i < NUM_TASKS; i++)
        cout << tasks[i]->result << ", ";
    
    cout << endl;
}

int main(int argc, char *argv[])
{
    ThreadGroup threads;
    MyRunTask *tasks[NUM_TASKS];

    try
    {
        
        for (int i = 0; i < NUM_TASKS; i++)
        {
            tasks[i] = new MyRunTask();
            threads.createThread(tasks[i]);
        }
        
        print(tasks);print(tasks);print(tasks);print(tasks);

        threads.joinAll();
        
        std::cout << "Finished all" << std::endl;
    }

    catch (except::Throwable& t)
    {
        cout << "Exception Caught: " << t.toString() << endl;
        return -1;
    }
    catch (...)
    {
        cout << "Exception Caught!" << endl;
        return -1;
    }

    return 0;
}
