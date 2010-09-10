/* =========================================================================
 * This file is part of net-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * net-c++ is free software; you can redistribute it and/or modify
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

#ifndef __NET_CLIENT_SOCKET_FACTORY_H__
#define __NET_CLIENT_SOCKET_FACTORY_H__

#include "net/Socket.h"

/*!
 *  \file
 *  \brief Contains routines for creation of client side sockets
 *
 *  The classes contained in this file are factory producers of
 *  client-side sockets.  The derived classes support basic UDP
 *  and TCP socket creation
 *
 */

namespace net
{
/*!
 *  \class ClientSocketFactory
 *  \brief Base class to create sockets
 *
 *  This class exists to create sockets of the given protocol.
 *  Possible protocols are defined in Socket.h
 *
 *  The derived classes specialize these for basic utility UDP
 *  and TCP sockets.
 *
 *
 */
class ClientSocketFactory
{
public:

    /*!
     *  Constructor.  Takes a protocol.  One of the notable
     *  behaviors of this constructor is that on Windows, it
     *  it attempts to register the winsock DLL, which it does
     *  using the net::Win32SocketInit() function.  This function,
     *  using an internal Mutex for synchronization, and a double checking
     *  entry makes the appropriate startup call to the DLL, and registers
     *  an exit-time shutdown hook for the DLL.
     *
     */

    ClientSocketFactory(int proto) : mProto(proto)
    {
        NATIVE_CLIENTSOCKET_INIT();
    }

    //!  Destructor
    virtual ~ClientSocketFactory()
    {}

    /*!
     *  Create a socket associated with the given address.
     *  \param address An address that we are interested in communicating
     *  with
     *
     *  \return A socket
     */
    virtual Socket create(const SocketAddress& address)
    {
        Socket s(mProto);

        // Make sure we're set up for broadcasting if necessary
        int on = 1;
        s.setOption(SOL_SOCKET, SO_BROADCAST, on);

        s.connect(address);
        return s;
    }
protected:
    int mProto;
};

/*!
 *  \class TCPClientSocketFactory
 *  \brief Class associated with TCP creation
 *
 *  This class creates TCP sockets only.
 */

class TCPClientSocketFactory : public ClientSocketFactory
{
public:
    TCPClientSocketFactory() : ClientSocketFactory(TCP_PROTO)
    {}

};

/*!
 *  \class UDPClientSocketFactory
 *  \brief Class associated with UDP creation
 *
 *  This class creates UDP sockets only.
 */

class UDPClientSocketFactory : public ClientSocketFactory
{
public:
    UDPClientSocketFactory() : ClientSocketFactory(UDP_PROTO)
    {}

};

}

#endif
