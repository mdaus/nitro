#ifndef __NET_DAEMON_WIN32_H__
#define __NET_DAEMON_WIN32_H__

#if defined(WIN32)

#include "net/DaemonInterface.h"
#include <import/except.h>

namespace net
{

/*!
 *  \class DaemonWin32
 *  \brief Windows implementation of daemon class
 */
class DaemonWin32 : public DaemonInterface
{
public:
    DaemonWin32() : DaemonInterface() {}

    void start()
    {
        throw except::NotImplementedException(
            Ctxt("Windows service not yet implemented."));
    }

    void stop()
    {
        throw except::NotImplementedException(
            Ctxt("Windows service not yet implemented."));
    }

    void restart()
    {
        throw except::NotImplementedException(
            Ctxt("Windows service not yet implemented."));
    }

    //! Parse and execute command line option (start/stop/restart)
    void daemonize(int& argc, char**& argv)
    {
        throw except::NotImplementedException(
            Ctxt("Windows service not yet implemented."));
    }

    void setTracefile(const std::string& tracefile) {}
    void setPidfile(const std::string& pidfile) {}
    std::string getTracefile() const { return ""; }
    std::string getPidfile() const { return ""; }
};

}

#endif
#endif
