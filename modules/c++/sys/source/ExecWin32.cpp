/* =========================================================================
 * This file is part of sys-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 *
 * sys-c++ is free software; you can redistribute it and/or modify
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


#if defined(WIN32)

#include <sys/Exec.h>
#include <str/Manip.h>

#define READ_PIPE  0
#define WRITE_PIPE 1

#include <fcntl.h>
#include <io.h>
typedef int pid_t;
#define pipe(phandles)  _pipe(phandles, 4096, _O_BINARY)
#define pclose _pclose
#define fileno _fileno

namespace sys
{

FILE* ExecPipe::openPipe(const std::string& command,
                         const std::string& type)
{
    register FILE* ioFile;
    HANDLE outIO[2] = {NULL, NULL};
    HANDLE inIO[2]  = { NULL, NULL };

    //! inherit the pipe handles
    SECURITY_ATTRIBUTES saAttr; 
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL; 
    if (!CreatePipe(&outIO[READ_PIPE], &outIO[WRITE_PIPE], &saAttr, 0))
    {
        return NULL;
    }
    if (!CreatePipe(&inIO[READ_PIPE], &inIO[WRITE_PIPE], &saAttr, 0))
    {
        return NULL;
    }

    // check the pipes themselves are not inherited
    if (!SetHandleInformation(outIO[READ_PIPE], HANDLE_FLAG_INHERIT, 0))
    {
        return NULL;
    }
    if (!SetHandleInformation(inIO[WRITE_PIPE], HANDLE_FLAG_INHERIT, 0))
    {
        return NULL;
    }

    // the startInfo structure is where the pipes are connected 
    ZeroMemory(&mProcessInfo, sizeof(PROCESS_INFORMATION));
    ZeroMemory(&mStartInfo, sizeof(STARTUPINFO));
    mStartInfo.cb = sizeof(STARTUPINFO); 
    mStartInfo.hStdInput = inIO[READ_PIPE];
    mStartInfo.hStdOutput = outIO[WRITE_PIPE];
    mStartInfo.hStdError = outIO[WRITE_PIPE];
    mStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    //! create the subprocess --
    //  this is equivalent to a fork + exec
    if (CreateProcessA(NULL, const_cast<char*>(command.c_str()),
                       NULL, NULL, TRUE, 0, NULL, NULL,
                       &mStartInfo, &mProcessInfo) == 0)
    {
        return NULL;
    }

    //  connect the pipes currently connected in the subprocess
    //  to the FILE* handle. Close the unwanted handle.
    if (type == "r")
    {
        int readDescriptor = 0;
        if ((readDescriptor = _open_osfhandle(
                (intptr_t)outIO[READ_PIPE], _O_RDONLY)) == -1)
        {
            return NULL;
        }
        ioFile = _fdopen(readDescriptor, type.c_str());
        CloseHandle(outIO[WRITE_PIPE]);
    }
    else
    {
        int writeDescriptor = 0;;
        if ((writeDescriptor = _open_osfhandle(
                (intptr_t)inIO[WRITE_PIPE], _O_WRONLY)) == -1)
        {
            return NULL;
        }
        ioFile = _fdopen(writeDescriptor, type.c_str());
        CloseHandle(inIO[READ_PIPE]);
    }

    return ioFile;
}

int ExecPipe::killProcess()
{
    //! issue a forceful removal of the process
    TerminateProcess(mProcessInfo.hProcess, PROCESS_TERMINATE);

    //! now clean up the process --
    //  wait needs to be called to remove the
    //  zombie process.
    return closePipe();
}

int ExecPipe::closePipe()
{
    if (!mOutStream)
    {
        throw except::IOException(
            Ctxt("The stream is already closed"));
    }

    // in case it fails
    FILE* tmp = mOutStream;
    mOutStream = NULL;

    DWORD dwMillisec = INFINITE;
    DWORD dwWaitStatus = 
        WaitForSingleObject(mProcessInfo.hProcess, dwMillisec);

    //! get the exit code
    DWORD exitCode = NULL;
    GetExitCodeProcess(mProcessInfo.hProcess, &exitCode);
    const int exitStatus = static_cast<int>(exitCode);
    if (exitStatus == -1)
    {
        sys::SocketErr err;
        throw except::IOException(
                Ctxt("Failure while closing stream to child process: " + 
                     err.toString()));

    }

    return exitStatus;
}

}

#endif
