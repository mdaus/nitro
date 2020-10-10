#pragma once

#pragma once

// Don't want to include all of GSL right now ...

#include <assert.h>

#include <exception>
#include <type_traits>
#include <utility>  

namespace gsl
{
    namespace details
    {
        [[noreturn]] inline void terminate() noexcept
        {
            std::terminate();
        }

        template <typename Exception>
        [[noreturn]] void throw_exception(Exception&&) noexcept
        {
            gsl::details::terminate();
        }
    }

    // narrow_cast(): a searchable way to do narrowing casts of values
    template <class T, class U> constexpr T narrow_cast(U&& u) noexcept
    {
        return static_cast<T>(std::forward<U>(u));
    }

    struct narrowing_error : public std::exception { };

    namespace details
    {
        template <class T, class U>
        struct is_same_signedness
            : public std::integral_constant<bool, std::is_signed<T>::value == std::is_signed<U>::value> { };
    } // namespace details

    // narrow() : a checked version of narrow_cast() that throws if the cast changed the value
    template <class T, class U>
    constexpr T narrow(U u) noexcept(false)
    {
        T t = narrow_cast<T>(u);
        if (static_cast<U>(t) != u) gsl::details::throw_exception(narrowing_error());
        if (!details::is_same_signedness<T, U>::value && ((t < T{}) != (u < U{})))
            gsl::details::throw_exception(narrowing_error());
        return t;
    }
}