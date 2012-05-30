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
		for (int count = 0; count < 1000000; count++)
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
        
        for (int i = 0; i < 5; i++)
        {
            print(tasks);
        }

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
