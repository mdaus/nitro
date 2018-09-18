/* =========================================================================
 * This file is part of CODA-OSS
 * =========================================================================
 *
 * (C) Copyright 2004 - 2018, MDA Information Systems LLC
 *
 * CODA-OSS is free software; you can redistribute it and/or modify
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

#include <mem/ScratchMemory.h>

#include <mem/BufferView.h>
#include <sys/Conf.h>
#include "TestCase.h"

namespace
{
TEST_CASE(testScratchMemory)
{
    mem::ScratchMemory scratch;

    TEST_ASSERT_EQ(scratch.getNumBytes(), 0);

    scratch.put<sys::ubyte>("buf0", 11, 1, 13);
    scratch.put<int>("buf1", 17, 1, 23);
    scratch.put<char>("buf2", 29, 3, 31);
    scratch.put<double>("buf3", 8);

    size_t numBytes0 = 11 + 13 - 1;
    size_t numBytes1 = 17 * sizeof(int) + 23 - 1;
    size_t numBytes2 = 3 * (29 + 31 - 1);
    size_t numBytes3 = 8 * sizeof(double) + sys::SSE_INSTRUCTION_ALIGNMENT - 1;
    TEST_ASSERT_EQ(scratch.getNumBytes(),
                   numBytes0 + numBytes1 + numBytes2 + numBytes3);

    // trying to get scratch before setting up should throw
    TEST_EXCEPTION(scratch.get<sys::ubyte>("buf0"));

    // set up external buffer
    std::vector<sys::ubyte> storage(scratch.getNumBytes());
    mem::BufferView<sys::ubyte> buffer(storage.data(), storage.size());

    // first pass with external buffer, second with internal allocation
    for (size_t ii = 0; ii < 2; ++ii)
    {
        if (ii == 0)
        {
            scratch.setup(buffer);
        }
        else
        {
            scratch.setup();
        }

        // trying to get nonexistent key should throw
        TEST_EXCEPTION(scratch.get<char>("buf999"));

        sys::ubyte* pBuf0 = scratch.get<sys::ubyte>("buf0");
        sys::ubyte* pBuf1 = scratch.get<sys::ubyte>("buf1");
        sys::ubyte* pBuf2_0 = scratch.get<sys::ubyte>("buf2", 0);
        sys::ubyte* pBuf2_1 = scratch.get<sys::ubyte>("buf2", 1);
        sys::ubyte* pBuf2_2 = scratch.get<sys::ubyte>("buf2", 2);
        sys::ubyte* pBuf3 = scratch.get<sys::ubyte>("buf3");

        // verify getBufferView matches get
        mem::BufferView<sys::ubyte> bufView0 =
                scratch.getBufferView<sys::ubyte>("buf0");
        mem::BufferView<sys::ubyte> bufView1 =
                scratch.getBufferView<sys::ubyte>("buf1");
        mem::BufferView<sys::ubyte> bufView2_0 =
                scratch.getBufferView<sys::ubyte>("buf2", 0);
        mem::BufferView<sys::ubyte> bufView2_1 =
                scratch.getBufferView<sys::ubyte>("buf2", 1);
        TEST_ASSERT_EQ(pBuf0, bufView0.data);
        TEST_ASSERT_EQ(pBuf1, bufView1.data);
        TEST_ASSERT_EQ(pBuf2_0, bufView2_0.data);
        TEST_ASSERT_EQ(pBuf2_1, bufView2_1.data);

        // verify getBufferView size in bytes
        TEST_ASSERT_EQ(bufView0.size, 11);
        TEST_ASSERT_EQ(bufView1.size, 17 * sizeof(int));
        TEST_ASSERT_EQ(bufView2_0.size, 29);
        TEST_ASSERT_EQ(bufView2_1.size, 29);

        // verify get works with const reference to ScratchMemory
        const mem::ScratchMemory& constScratch = scratch;
        const sys::ubyte* pConstBuf0 = constScratch.get<sys::ubyte>("buf0");
        TEST_ASSERT_EQ(pBuf0, pConstBuf0);

        // verify getBufferView works with const reference to ScratchMemory
        mem::BufferView<const sys::ubyte> constBufView0 =
                constScratch.getBufferView<sys::ubyte>("buf0");
        TEST_ASSERT_EQ(bufView0.data, constBufView0.data);

        // trying to get buffer index out of range should throw
        TEST_EXCEPTION(scratch.get<sys::ubyte>("buf0", 1));
        TEST_EXCEPTION(scratch.get<sys::ubyte>("buf0", -1));
        TEST_EXCEPTION(scratch.get<sys::ubyte>("buf2", 3));

        // verify alignment
        TEST_ASSERT_EQ(reinterpret_cast<size_t>(pBuf0) % 13, 0);
        TEST_ASSERT_EQ(reinterpret_cast<size_t>(pBuf1) % 23, 0);
        TEST_ASSERT_EQ(reinterpret_cast<size_t>(pBuf2_0) % 31, 0);
        TEST_ASSERT_EQ(reinterpret_cast<size_t>(pBuf2_1) % 31, 0);
        TEST_ASSERT_EQ(reinterpret_cast<size_t>(pBuf2_2) % 31, 0);
        TEST_ASSERT_EQ(reinterpret_cast<size_t>(pBuf3) % sys::SSE_INSTRUCTION_ALIGNMENT, 0);

        // verify no overlap between buffers
        TEST_ASSERT_TRUE(pBuf1 - pBuf0 >= static_cast<ptrdiff_t>(11));
        TEST_ASSERT_TRUE(pBuf2_0 - pBuf1 >=
                static_cast<ptrdiff_t>(17 * sizeof(int)));
        TEST_ASSERT_TRUE(pBuf2_1 - pBuf2_0 >= static_cast<ptrdiff_t>(29));
        TEST_ASSERT_TRUE(pBuf2_2 - pBuf2_1 >= static_cast<ptrdiff_t>(29));
        TEST_ASSERT_TRUE(pBuf3 - pBuf2_2 >= static_cast<ptrdiff_t>(29));
    }

    // put should invalidate the scratch memory until setup is called again
    scratch.put<sys::ubyte>("buf4", 8);
    TEST_EXCEPTION(scratch.get<sys::ubyte>("buf0"));
    scratch.setup();
    scratch.get<sys::ubyte>("buf0");

    // calling setup with buffer that is too small should throw
    mem::BufferView<sys::ubyte> smallBuffer(buffer.data, buffer.size - 1);
    TEST_EXCEPTION(scratch.setup(smallBuffer));

    // calling setup with invalid external buffer should throw
    mem::BufferView<sys::ubyte> invalidBuffer(NULL, buffer.size);
    TEST_EXCEPTION(scratch.setup(invalidBuffer));
}
}

int main(int, char**)
{
    TEST_CHECK(testScratchMemory);

    return 0;
}
