/* =========================================================================
 * This file is part of str-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 *
 * str-c++ is free software; you can redistribute it and/or modify
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

#include <assert.h>
#include <string.h> // strlen()
#include <wchar.h>

#include <map>
#include <locale>
#include <stdexcept>

#include "str/Convert.h"
#include "str/Manip.h"
#include "str/utf8.h"

template<> std::string str::toType<std::string>(const std::string& s)
{
    return s;
}

template<> std::string str::toString(const uint8_t& value)
{
    return str::toString(static_cast<unsigned int>(value));
}

template<> std::string str::toString(const int8_t& value)
{
    return str::toString(static_cast<int>(value));
}

template<> bool str::toType<bool>(const std::string& s)
{
    std::string ss = s;
    str::lower(ss);

    if (ss == "true")
    {
        return true;
    }
    else if (ss == "false")
    {
        return false;
    }
    else if (str::isNumeric(ss))
    {
        int value(0);
        std::stringstream buf(ss);
        buf >> value;
        return (value != 0);
    }
    else
    {
        throw except::BadCastException(except::Context(__FILE__, __LINE__,
            std::string(""), std::string(""),
            std::string("Invalid bool: '") + s + std::string("'")));
    }

    return false;
}

long long str::strtoll(const char *str, char **endptr, int base)
{
#if defined(_MSC_VER)
    return _strtoi64(str, endptr, base);
#else
    return ::strtoll(str, endptr, base);
#endif
}

unsigned long long str::strtoull(const char *str, char **endptr, int base)
{
#if defined(_MSC_VER)
    return _strtoui64(str, endptr, base);
#else
    return ::strtoull(str, endptr, base);
#endif
}

template<> int str::getPrecision(const float& )
{
    return std::numeric_limits<float>::max_digits10;
}
template<> int str::getPrecision(const double& )
{
    return std::numeric_limits<double>::max_digits10;
}
template<> int str::getPrecision(const long double& )
{
    return std::numeric_limits<long double>::max_digits10;
}

// Need to look up characters from \x80 (EURO SIGN) to \x9F (LATIN CAPITAL LETTER Y WITH DIAERESIS)
// in a map: http://www.unicode.org/Public/MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1252.TXT
static inline str::U8string utf8_(uint32_t ch)
{
    const std::u32string s{static_cast<std::u32string::value_type>(ch)};
    str::U8string retval;
    str::utf32to8(s, retval);
    return retval;
};
static const std::map<uint32_t, sys::U8string> Windows1252_x80_x9F_to_u8string{
    {0x80, utf8_(0x20AC) } // EURO SIGN
    // , {0x81, replacement_character } // UNDEFINED
    , {0x82, utf8_(0x201A) } // SINGLE LOW-9 QUOTATION MARK
    , {0x83, utf8_(0x0192)  } // LATIN SMALL LETTER F WITH HOOK
    , {0x84, utf8_(0x201E)  } // DOUBLE LOW-9 QUOTATION MARK
    , {0x85, utf8_(0x2026)  } // HORIZONTAL ELLIPSIS
    , {0x86, utf8_(0x2020)  } // DAGGER
    , {0x87, utf8_(0x2021)  } // DOUBLE DAGGER
    , {0x88, utf8_(0x02C6)  } // MODIFIER LETTER CIRCUMFLEX ACCENT
    , {0x89, utf8_(0x2030)  } // PER MILLE SIGN
    , {0x8A, utf8_(0x0160)  } // LATIN CAPITAL LETTER S WITH CARON
    , {0x8B, utf8_(0x2039)  } // SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    , {0x8C, utf8_(0x0152)  } // LATIN CAPITAL LIGATURE OE
    //, {0x8D, replacement_character } // UNDEFINED
    , {0x8E, utf8_(0x017D)  } // LATIN CAPITAL LETTER Z WITH CARON
    //, {0x8F, replacement_character } // UNDEFINED
    //, {0x90, replacement_character } // UNDEFINED
    , {0x91, utf8_(0x017D)  } // LEFT SINGLE QUOTATION MARK
    , {0x92, utf8_(0x2018)  } // RIGHT SINGLE QUOTATION MARK
    , {0x93, utf8_(0x2019)  } // LEFT DOUBLE QUOTATION MARK
    , {0x94, utf8_(0x201D)  } // RIGHT DOUBLE QUOTATION MARK
    , {0x95, utf8_(0x2022)  } // BULLET
    , {0x96, utf8_(0x2013)  } // EN DASH
    , {0x97, utf8_(0x2014)  } // EM DASH
    , {0x98, utf8_(0x02DC)  } // SMALL TILDE
    , {0x99, utf8_(0x2122)  } // TRADE MARK SIGN
    , {0x9A, utf8_(0x0161)  } // LATIN SMALL LETTER S WITH CARON
    , {0x9B, utf8_(0x203A)  } // SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    , {0x9C, utf8_(0x0153)  } // LATIN SMALL LIGATURE OE
    //, {0x9D, replacement_character } // UNDEFINED
    , {0x9E, utf8_(0x017E)  } // LATIN SMALL LETTER Z WITH CARON
    , {0x9F, utf8_(0x0178)  } // LATIN CAPITAL LETTER Y WITH DIAERESIS
};

// Convert a single Windows-1252 character to UTF-8
// https://en.wikipedia.org/wiki/ISO/IEC_8859-1
static constexpr sys::U8string::value_type cast(uint8_t ch)
{
    static_assert(sizeof(decltype(ch)) == sizeof(sys::U8string::value_type), "sizeof(uint8_t) != sizeof(Char8_t)");
    return static_cast<sys::U8string::value_type>(ch);
}
static sys::U8string fromWindows1252(uint8_t ch)
{
    // ASCII is the same in UTF-8
    if (ch < 0x80)
    {
        return sys::U8string{cast(ch)};  // ASCII
    }

    // ISO8859-1 can be converted to UTF-8 with bit-twiddling
    if (ch > 0x9F)
    {
        // https://stackoverflow.com/questions/4059775/convert-iso-8859-1-strings-to-utf-8-in-c-c
        // *out++=0xc2+(*in>0xbf), *out++=(*in++&0x3f)+0x80;
        return sys::U8string{cast(0xc2 + (ch > 0xbf)), cast((ch & 0x3f) + 0x80)}; // ISO8859-1
    }

    static const auto map = Windows1252_x80_x9F_to_u8string;
    const auto it = map.find(ch);
    if (it != map.end())
    {
        return it->second;
    }

    // If the input text contains a character that isn't defined in Windows-1252;
    // return a "replacment character."  Yes, this will **corrupt** the input data as information is lost:
    // https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character
    static const sys::U8string replacement_character = utf8_(0xfffd);
    return replacement_character;
}
inline sys::U8string fromWindows1252(std::string::value_type ch_)
{
    return fromWindows1252(static_cast<uint8_t>(ch_));
}
inline sys::U8string fromWindows1252(str::W1252string::value_type ch_)
{
    return fromWindows1252(static_cast<uint8_t>(ch_));
}

void str::windows1252to8(W1252string::const_pointer p, size_t sz, sys::U8string& result)
{
    for (size_t i = 0; i < sz; i++)
    {
        result += ::fromWindows1252(p[i]);    
    }
}

static std::map<std::u32string::value_type, std::string::value_type> make_u32string_to_Windows1252()
{
    std::map<std::u32string::value_type, std::string::value_type> retval;

    const auto& m = Windows1252_x80_x9F_to_u8string;
    for (const auto& p : m)
    {
        const auto ch_utf32 = p.first;
        const auto& str_utf8 = p.second;

        std::wstring str_wc;
        if (mbtowc(str_utf8, str_wc))
        {
            retval[str_wc[0]] = static_cast<char>(ch_utf32);
        }
    }

    return retval;
}
static std::string::value_type toWindows1252(std::u32string::value_type ch)
{
    // ASCII
    if (ch < 0x00000080)
    {
        return static_cast <std::string::value_type>(ch);
    }

    static const auto map = make_u32string_to_Windows1252();
    const auto it = map.find(ch);
    if (it != map.end())
    {
        return it->second;
    }

    // a wide-character that can't be converted to Windows-1252
    if (ch > 0x000000ff)
    {
        // This will **corrupt** the input data as information is lost.
        // https://en.wikipedia.org/wiki/Windows-1252
        return static_cast <std::string::value_type>(0x81); // UNDEFINED
    }

    return static_cast<std::string::value_type>(ch);
}

template<typename T>
inline void strto1252_(const T& p, size_t sz, str::W1252string& result)
{
    for (size_t i = 0; i < sz; i++)
    {
        result += static_cast<str::W1252string::value_type>(::toWindows1252(p[i]));
    }
}
void str::utf16to1252(std::u16string::const_pointer p, size_t sz, W1252string& result)
{
    strto1252_(p, sz, result);
}
void str::utf32to1252(std::u32string::const_pointer p, size_t sz, W1252string& result)
{
    strto1252_(p, sz, result);
}
void str::wsto1252(std::wstring::const_pointer p, size_t sz, W1252string& result)
{
    strto1252_(p, sz, result);
}

namespace
{
void utf16to8(const std::u16string& str, std::string& result)
{
    // http://utfcpp.sourceforge.net/#introsample
    /*
    // Convert it to utf-16
    vector<unsigned short> utf16line;
    utf8::utf8to16(line.begin(), end_it, back_inserter(utf16line));

    // And back to utf-8
    string utf8line;
    utf8::utf16to8(utf16line.begin(), utf16line.end(), back_inserter(utf8line));
    */
    utf8::utf16to8(str.begin(), str.end(), std::back_inserter(result));
}
void utf32to8(const std::u32string& str, std::string& result)
{
    utf8::utf32to8(str.begin(), str.end(), std::back_inserter(result));
}
}

// What is the corresponding std::uXXstring for std::wstring?
template<size_t sizeof_wchar_t> struct const_pointer final { };
template<> struct const_pointer<2> // wchar_t is 2 bytes, UTF-16
{
    using type = std::u16string::const_pointer;
};
template <> struct const_pointer<4> // wchar_t is 4 bytes, UTF-32
{
    using type = std::u32string::const_pointer;
};
inline void wsto8_(std::u16string::const_pointer begin, std::u16string::const_pointer end, std::string& result)
{
    utf8::utf16to8(begin, end, std::back_inserter(result));
}
inline void wsto8_(std::u32string::const_pointer begin, std::u32string::const_pointer end, std::string& result)
{
    utf8::utf32to8(begin, end, std::back_inserter(result));
}
namespace
{
void wsto8(const std::wstring& str, std::string& result)
{
    // std::wstring is UTF-16 on Windows, UTF-32 on Linux
    using const_pointer_t = const_pointer<sizeof(std::wstring::value_type)>::type;
    auto const begin = str::c_str<const_pointer_t>(str);
    wsto8_(begin, begin + str.size(), result);
}
}

struct back_inserter final
{ 
    sys::U8string* container = nullptr; // pointer instead of reference for copy
    explicit back_inserter(sys::U8string& s) noexcept : container(&s) { }

    back_inserter& operator=(uint8_t v)
    {
        container->push_back(static_cast<sys::U8string::value_type>(v));
        return *this;
    }
    back_inserter& operator*() noexcept { return *this; }
    back_inserter operator++(int) noexcept { return *this; }
};
void str::utf16to8(std::u16string::const_pointer p, size_t sz, sys::U8string& result)
{
    utf8::utf16to8(p, p + sz, back_inserter(result));
}
void str::utf32to8(std::u32string::const_pointer p, size_t sz, sys::U8string& result)
{
    utf8::utf32to8(p, p + sz, back_inserter(result));
}

inline void wsto8_(std::u16string::const_pointer begin, std::u16string::const_pointer end, sys::U8string& result)
{
    utf8::utf16to8(begin, end, back_inserter(result));
}
inline void wsto8_(std::u32string::const_pointer begin, std::u32string::const_pointer end, sys::U8string& result)
{
    utf8::utf32to8(begin, end, back_inserter(result));
}
void str::wsto8(std::wstring::const_pointer p, size_t sz, sys::U8string& result)
{
    using const_pointer_t = const_pointer<sizeof(*p)>::type;
    const void* const pStr = p;
    auto const begin = static_cast<const_pointer_t>(pStr);
    wsto8_(begin, begin + sz, result);
}

str::setlocale::setlocale(const char* locale) : locale_(::setlocale(LC_ALL, locale))
{
    if (locale_ == nullptr)
    {
        throw std::runtime_error("setlocale() failed.");
    }
}
str::setlocale::~setlocale() noexcept(false)
{
    if (::setlocale(LC_ALL, locale_) == nullptr)
    {
        throw std::runtime_error("setlocale() failed.");       
    }
}

template<typename TString>
bool fromWindows1252_(const TString& s, std::wstring& result)
{
    auto const p = str::c_str<str::W1252string::const_pointer>(s);

    sys::U8string utf8;
    str::windows1252to8(p, s.size(), utf8);
    return str::mbtowc(utf8, result);
}
template<typename TString>
bool fromUTF8_(const TString& s, std::wstring& result)
{
    auto const p = str::c_str<sys::U8string::const_pointer>(s);
    return str::mbtowc(p, s.size(), result);
}

std::wstring str::to_wstring(const sys::U8string& s)
{
    std::wstring retval;
    if (!fromUTF8_(s, retval))
    {
        throw std::runtime_error("mbtowc() failed.");
    }
    return retval;
}
std::wstring to_wstring(const str::W1252string& s)
{
    std::wstring retval;
    if (!fromWindows1252_(s, retval))
    {
        throw std::runtime_error("mbtowc() failed.");
    }
    return retval;
}
std::wstring str::to_wstring(const std::string& s)
{
    std::wstring retval;
    const auto result =
    #ifdef _WIN32
    fromWindows1252_(s, retval);
    #else
    fromUTF8_(s, retval);
    #endif

    if (!result)
    {
        throw std::runtime_error("mbtowc() failed.");
    }
    return retval;
}

sys::U8string str::to_u8string(const std::wstring& s)
{
    sys::U8string retval;
    if (!wctomb(s, retval))
    {
        throw std::runtime_error("wctomb() failed.");
    }

    #ifndef NDEBUG
    // Double-check against utf8:: code
    sys::U8string retval2;
    strto8(s, retval2);
    assert(retval == retval2);
    #endif

    return retval;
}
sys::U8string str::to_u8string(const W1252string& s)
{
    sys::U8string retval;
    windows1252to8(s.c_str(), s.size(), retval);
    return retval;
}
sys::U8string str::to_u8string(const std::string& s)
{
    sys::U8string retval;
#ifdef _WIN32
    windows1252to8(str::c_str<W1252string::const_pointer>(s), s.size(), retval);
#else
    retval = return c_str<sys::U8string::const_pointer>(s);  // copy
#endif

    return retval;
}

bool str::mbtowc(sys::U8string::const_pointer in_, size_t in_sz, std::wstring& result)
{
    const setlocale utf8_locale;

    // This is OK as UTF-8 can be stored in std::string
    // Note that casting between the string types will CRASH on some
    // implementatons. NO: reinterpret_cast<const std::string&>(value)
    const void* const pValue = in_;

    // https://en.cppreference.com/w/c/string/multibyte/mbrtowc
    mbstate_t state;
    memset(&state, 0, sizeof(state));

    auto const in = static_cast<std::string::const_pointer>(pValue);
 
    std::vector<std::wstring::value_type> v_out(in_sz+1);
    auto const out = v_out.data();
    const char *p_in = in, *end = in + in_sz;
    wchar_t* p_out = out;
    size_t rc;
    while ((rc = mbrtowc(p_out, p_in, end - p_in, &state)) > 0)
    {
        p_in += rc;
        p_out += 1;
    }
    const auto rc_ = static_cast<ptrdiff_t>(rc);
    if ((rc_ < 0) && (rc_ != -2)) // "if the next n bytes constitute an incomplete, but so far valid, multibyte character. Nothing is written to *pwc."
    {
        // https://en.cppreference.com/w/cpp/string/multibyte/mbrtowc
        // "if encoding error occurs. Nothing is written to *pwc, the value EILSEQ is stored in errno and the value of *ps is left unspecified."
        return false;
    }
    const size_t out_sz = p_out - out;
    result = std::wstring(out, out_sz);  // UTF-16 on Windows, UTF-32 on Linux
    return true;
}
bool str::mbtowc(const sys::U8string& utf8, std::wstring& result)
{
    return mbtowc(utf8.c_str(), utf8.size(), result);
}

bool str::wctomb(std::wstring::const_pointer in, size_t in_sz, sys::U8string& result)
{
    const setlocale utf8_locale;

    // https://en.cppreference.com/w/c/string/multibyte/wcrtomb
    mbstate_t state;
    memset(&state, 0, sizeof(state));

    std::vector<sys::U8string::value_type> v_out(MB_CUR_MAX * (in_sz+1));
    void* const out_ = v_out.data();
    auto const out = static_cast<char*>(out_);
    auto p = out;
    size_t rc = 0;
    ptrdiff_t rc_ = 0;
    for (size_t n = 0; n < in_sz; ++n)
    {
        #if _MSC_VER
        #pragma warning(push)
        #pragma warning(disable: 4996) // may be unsafe
        #endif
        rc = wcrtomb(p, in[n], &state);
        #if _MSC_VER
        #pragma warning(pop)
        #endif
         rc_ = static_cast<ptrdiff_t>(rc);

        if (rc_ == -1)
            break;
        p += rc;
    }
    if (rc_ < 0)
    {
        // https://en.cppreference.com/w/c/string/multibyte/wcrtomb
        return false;
    }

    const size_t out_sz = p - out;
    result = sys::U8string(v_out.data(), out_sz);
    return true;
}
bool str::wctomb(const std::wstring& s, sys::U8string& result)
{
    return wctomb(s.c_str(), s.size(), result);
}

namespace
{
bool toWindows1252_(const std::wstring& s, std::string& result)
{
    str::W1252string w1252;
    str::wsto1252(s.c_str(), s.size(), w1252);

    auto const pStr = str::c_str<std::string::const_pointer>(w1252);
    result = pStr;  // copy
    return true;
}
bool toUTF8_(const std::wstring& s, std::string& result)
{
    str::U8string utf8;
    if (str::wctomb(s, utf8))
    {
        auto const pStr = str::c_str<std::string::const_pointer>(utf8);
        result = pStr;  // copy
        return true;
    }
    return false;
}
}
std::string str::to_string(const std::wstring& s)
{
    std::string retval;
    const auto result = 
    #ifdef _WIN32
        toWindows1252_(s, retval);
    #else
        toUTF8_(s, retval);
    #endif
    if (!result)
    {
        throw std::runtime_error("wctomb() failed.");
    }
    return retval;
}
