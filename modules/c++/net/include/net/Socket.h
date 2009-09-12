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

#ifndef __NET_SOCKET_H__
#define __NET_SOCKET_H__

#include "net/Sockets.h"
#include "net/SocketAddress.h"
#include "sys/SystemException.h"
#include "except/Exception.h"

/*!
 *  \file
 *  \brief File containing Socket class
 *
 *  This file contains all of the material necessary to use low-level
 *  socket functionality.
 *
 */


namespace net
{
//!  Supported protocols
enum { TCP_PROTO = SOCK_STREAM, UDP_PROTO = SOCK_DGRAM };

/*!
 *  \class Socket
 *  \brief Class for making socket API calls on either client or server
 *
 *  This class is the lowest functional API level for talking over sockets.
 *  It provides all of the calls necesary to establish client or server
 *  side UDP or TCP sockets.  Not all of these calls provided here can
 *  be used in all contexts.  It is up to the developer to have a basic
 *  knowledge of which functions to use in which cases.  Higher level
 *  users may find the functionality of NetConnection more to their liking.
 *
 */
class Socket
{
public:
    /*!
     *  Default constructor.  Does nothing.  Socket is left in
     *  invalid handle state
     */
    Socket() : mNative(INVALID_SOCKET)
    {}

    /*!
     *  Give a protocol
     *
     */
    Socket(int proto);

    /*!
     *  Copy constructor
     */
    Socket(const Socket& socket)
    {
        mNative = socket.mNative;
    }

    /*!
     *  Assignment operator
     *  \param socket
     */
    Socket& operator=(const Socket& socket)
    {
        if (&socket != this)
        {
            mNative = socket.mNative;
        }
        return *this;
    }

    /*!
     *  Destructor
     */
    virtual ~Socket()
    { }

    /*!
     *  Method to open a socket.  Copies a handle to its internal
     *  matter.  Connects the handles to their readers and writers.
     *  \param socket The handle to initialize 
     *  \throw SocketCreationFailedException
     */
    virtual void open(const Socket& socket)
    {
        mNative = socket.mNative;
    }

    /*!
     *  Close a socket.  This releases the writers/readers and closes
     *  the handle.
     */
    virtual void close()
    {
        closesocket(mNative);
    }

    /*!
     *  Convert to a passive socket.  Set maximum incoming queue
     *  from backlog.
     *
     *  \param backlog
     */
    virtual void listen(int backlog);

    /*!
     *  Connect the socket to this address.  This is usually called
     *  by the creator pattern.
     * 
     *  \param address The address to connect to
     */
    virtual void connect(const SocketAddress& address);

    /*!
     *  Bind the socket to this address.  This is usually called
     *  by the creator pattern.
     *
     */
    virtual void bind(const SocketAddress& address);


    /*!
     *  This method is simply a template overload of setsockopt(),
     *  established to make the routine a little less redundant.
     *
     *  \param level The level for the option (e.g, SOL_SOCKET)
     *  \param option The option value (e.g., SO_DEBUG)
     *  \param val The value of the option
     */
    template<typename T> void setOption(int level,
                                        int option,
                                        const T& val)
    {
        if (NATIVE_SOCKET_FAILED(::setsockopt(mNative, level, option, (const char*)&val,
                (net::SockLen_T)sizeof(T))))
        {
            
#if defined(WIN32) || defined(_WIN32)

       /* Wrapper for setsockopt dealing with Windows specific issues :-
             *
             * IP_TOS and IP_MULTICAST_LOOP can't be set on some Windows
             * editions. 
             * 
             * The value for the type-of-service (TOS) needs to be masked
             * to get consistent behaviour with other operating systems.
             */
            
       /*
             * IP_TOS & IP_MULTICAST_LOOP can't be set on some versions
             * of Windows.
             */
            if ((WSAGetLastError() == WSAENOPROTOOPT) && (level == IPPROTO_IP) &&
                    (option == IP_TOS || option == IP_MULTICAST_LOOP))
                return;

        /*
              * IP_TOS can't be set on unbound UDP sockets.
              */
            if ((WSAGetLastError() == WSAEINVAL) && (level == IPPROTO_IP) &&
                    (option == IP_TOS))
                return;
#endif 
        
            throw sys::SocketException(Ctxt("setsockop() function"));
        }
    }

    /*!
     *  This method is simply a template overload of getsockopt()
     *  established to make the routine less redundant
     *  \param level The level for the option (e.g, SOL_SOCKET)
     *  \param option The option value (e.g., SO_DEBUG)
     *  \param val The value of the option 
     *
     */
    template<typename T> void getOption(int level,
                                        int option,
                                        T& val)
    {
        net::SockLen_T size = (net::SockLen_T)sizeof(T);
        ::getsockopt(mNative, level, option, &val, &size);
    }

    /*!
     *  Socket read functionality
     *  \param b The byte buffer to recv into
     *  \param len The number of bytes to read
     *  \param flags (optional) Addtional flags (not common)
     */
    virtual sys::SSize_T recv(sys::byte* b,
                              sys::Size_T len,
                              int flags = 0);

    /*!
     *  Same as recv, except from a specified socket address.  Only
     *  really makes sense for UDP sockets.  The address recv'd from
     *  will be in the parameter, passed by reference
     *
     *  \param address The address we recv'd from
     *  \param b The bytes to read
     *  \param len The number of bytes read
     *  \param flags The flags (usually not specified)
     */

    virtual sys::SSize_T recvFrom(SocketAddress& address,
                                  sys::byte* b,
                                  sys::Size_T len,
                                  int flags = 0);


    /*!
     *  Send bytes over the internet
     *  \param b The byte buffer to send
     *  \param len The number of bytes.
     *  \param flags The flags (usually not specified)
     */
    virtual void send(const sys::byte* b,
                      sys::Size_T len,
                      int flags = 0);

    /*!
     *  Same as send, except to a specified socket address.  Only
     *  really makes sense for UDP sockets.  The address sent from
     *  will be in the parameter
     *
     *  \param address Address to send packet to
     *  \param b The bytes to read
     *  \param len The number of bytes read
     *  \param flags The flags (usually not specified)
     */

    virtual void sendTo(const SocketAddress& address,
                        const sys::byte* b,
                        sys::Size_T len,
                        int flags = 0);

    /*!
     *  Accept a connection while listening on a passive socket.
     *  Produces a connection to the client at the socket address.
     *  The parameter is actually a return value from the accept()
     *  call, and can be ignored completely if desired.
     *
     *  \param fromClient Client socket address returned
     *  \return A new socket connection to the client
     */
    virtual Socket accept(SocketAddress& fromClient);

    net::Socket_T getHandle() const
    {
        return mNative;
    }
protected:
    //! The socket
    net::Socket_T mNative;
};

}

#endif

