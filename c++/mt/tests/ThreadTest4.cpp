/* =========================================================================
 * This file is part of mt-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * mt-c++ is free software; you can redistribute it and/or modify
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

#include <import/mt.h>

using namespace mt;
using namespace std;

class Getter : public mt::Runnable
{
public:
    Getter(mt::Mutex *by, int * val, int n) : theVal(val), syncBy(by), id(n)
    {}
    virtual ~Getter()
    {}

    virtual void run()
    {
        for (int i = 0; i < 250; i++)
        {

            std::cout << "Getter::run: " << std::endl;
            std::cout << typeid(this).name() << std::endl;
            syncBy->lock();
            int x = get();
            cout << "Thread id: "<< id << " got back " << x << endl;
            syncBy->unlock();
            mt::Thread::yield();
        }
    }
    int get()
    {

        return *theVal;
    }
protected:
    int *theVal;
    mt::Mutex *syncBy;
    int id;

};
class Putter : public mt::Runnable
{
public:
    Putter(mt::Mutex *by,int *val, int n) : theVal(val), syncBy(by), id(n)
    {}
    virtual ~Putter()
    {}

    virtual void run()
    {

        std::cout << "Putter::run: " << std::endl;
        std::cout << typeid(this).name() << std::endl;

        for (int i = 0; i < 250; i++)
        {
            syncBy->lock();
            set(i);
            cout << "Thread id: "<< id << " set to " << i << endl;
            syncBy->unlock();

            mt::Thread::yield();

        }

    }
    void set(int val)
    {
        *theVal = val;
    }
protected:
    int *theVal;
    mt::Mutex *syncBy;
    int id;
};

int main()
{
    try
    {
        int val = 24;
        mt::Mutex syncBy;
        mt::Thread *gT[5];
        mt::Thread *pT[5];

        for (int i = 0; i < 5; i++)
        {

            gT[i] = new mt::Thread(new Getter(&syncBy, &val, i));
            gT[i]->start();

            pT[i] = new mt::Thread(new Putter(&syncBy, &val, i));
            pT[i]->start();

            // 	    //printf("p (&) %x\n", p);
            // 	    mt::Thread(p).start();
            // 	    mt::Thread(new Putter(&syncBy, &val, i)).start();
        }

        for (int i = 0; i < 5; i++)
        {
            gT[i]->join();
            cout << "Joined on gT[" << i << "]" << endl;
            delete gT[i];
            pT[i]->join();
            delete pT[i];
            cout << "Joined on pT[" << i << "]" << endl;
        }
        //	mt::Thread::yield();

    }
    catch (except::Exception& e)
    {
        cout << "Caught Exception: " << e.toString() << endl;
    }
    catch (...)
    {
        cout << "Unknown exception" << endl;
    }
    return 0;
};
