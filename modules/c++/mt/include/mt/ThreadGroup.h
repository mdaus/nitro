#ifndef __MT_THREAD_GROUP_H__
#define __MT_THREAD_GROUP_H__

#include <vector>
#include <memory>
#include "sys/Runnable.h"
#include "sys/Thread.h"

namespace mt
{
class ThreadGroup
{
public:
    ThreadGroup();
    ~ThreadGroup();
    void createThread(sys::Runnable *thread);
    void joinAll();
private:
    std::vector<sys::Thread*> threads;
    
    void createThread(std::auto_ptr<sys::Runnable> thread);
};
}

#endif
