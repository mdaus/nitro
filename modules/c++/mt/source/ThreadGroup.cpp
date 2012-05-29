#include "mt/ThreadGroup.h"

mt::ThreadGroup::ThreadGroup()
{}

mt::ThreadGroup::~ThreadGroup()
{
    try
    {
        joinAll();
    }
    catch (...)
    {
        // Make sure we don't throw out of the destructor.
    }
    
    for (int i = 0; i < threads.size(); i++)
        delete threads[i];
}

void mt::ThreadGroup::createThread(sys::Runnable *runnable)
{
    std::auto_ptr<sys::Runnable> thread_ptr(runnable);
    createThread(thread_ptr);
}

void mt::ThreadGroup::createThread(std::auto_ptr<sys::Runnable> runnable)
{
    sys::Thread *thread = new sys::Thread(runnable.get());
    threads.push_back(thread);
    thread->start();
	runnable.release();
}

void mt::ThreadGroup::joinAll()
{
    sys::Thread *thread;
    
    while (!threads.empty())
    {
        thread = threads.back();
        thread->join();
        delete thread;
        threads.pop_back();
    }
}
