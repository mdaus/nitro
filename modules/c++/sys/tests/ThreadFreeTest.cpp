#include <iostream>
#include <import/sys.h>
using namespace sys;
using namespace std;

class MyRunTask : public Runnable
{
public:
    int result;
    
    MyRunTask()
    {
        result = 0;
    }
    virtual ~MyRunTask()
    {
    }

    virtual void run()
    {
		result = 1;
    }
};

int main(int argc, char *argv[])
{
    Thread *thread;
    MyRunTask *task1;
    MyRunTask *task2;
    MyRunTask *task3;

    try
    {
        task1 = new MyRunTask();
        thread = new Thread(task1);
        thread->start();
        thread->join();
        task2 = new MyRunTask();
        
        if (task1->result != 1)
        {
            cout << "Task1 not run, result: " << task1->result << endl;
            return -1;
        }
        
        delete thread;

        task3 = new MyRunTask();
        
        if (task1 == task3)
            cout << "Task1 freed" << endl;
            
        
        delete task2;
        delete task3;
        
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
