#ifndef RTHMM_UTILS
#define RTHMM_UTILS

#include <memory>

#define RETURN(x) -> decltype(x) { return x; }

namespace rtHMM {

    namespace internal {
        template<typename T, typename ...Args>
        std::unique_ptr<T> make_unique(Args&& ...args)
        {
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
        }

        template<typename T, typename... Rest>
        struct are_same : std::true_type {};

        template<typename T, typename First>
        struct are_same<T, First> : std::is_same<T, First> {};

        template<typename T, typename First, typename... Rest>
        struct are_same<T, First, Rest...>
                : std::integral_constant<bool, std::is_same<T, First>::value&& are_same<T, Rest...>::value> {
        };
    }
}


#endif // RTHMM_UTILS
