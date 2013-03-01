#ifndef __MT_RUNNABLE_1D_H__
#define __MT_RUNNABLE_1D_H__

#include <sys/Runnable.h>
#include "mt/ThreadPlanner.h"
#include "mt/ThreadGroup.h"

namespace mt
{
template <typename OpT>
class Runnable1D : public sys::Runnable
{
public:
    Runnable1D(size_t startElement,
               size_t numElements,
               const OpT& op) :
        mStartElement(startElement),
        mEndElement(startElement + numElements),
        mOp(op)
    {
    }

    virtual void run()
    {
        for (size_t ii = mStartElement; ii < mEndElement; ++ii)
        {
            mOp(ii);
        }
    }

private:
    const size_t mStartElement;
    const size_t mEndElement;
    const OpT& mOp;
};

template <typename OpT>
void run1D(size_t numElements, size_t numThreads, const OpT& op)
{
    if (numThreads <= 1)
    {
        Runnable1D<OpT>(0, numElements, op).run();
    }
    else
    {
        ThreadGroup threads;
        const ThreadPlanner planner(numElements, numThreads);
 
        size_t threadNum(0);
        size_t startElement(0);
        size_t numElementsThisThread(0);
        while(planner.getThreadInfo(threadNum++, startElement, numElementsThisThread))
        {
            threads.createThread(new Runnable1D<OpT>(
                startElement, numElementsThisThread, op));
        }
        threads.joinAll();
    }
}
}

#endif
