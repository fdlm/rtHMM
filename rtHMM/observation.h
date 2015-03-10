#ifndef RTHMM_OBSERVATION_H
#define RTHMM_OBSERVATION_H

#include <typeinfo>

namespace rtHMM {

    class observation {
        public:
            template<typename T>
            observation(const T& obs);

            template<typename T>
            const T& safe_cast() const;

            template<typename T>
            bool has_type() const;

            template<typename T>
            const T& cast() const;

        private:
            const void* obs;
            const std::type_info& obs_type;
    };

    template<typename T>
    observation::observation(const T &obs) :
        obs(reinterpret_cast<const void*>(&obs)),
        obs_type(typeid(T))
    {
    }

    template<typename T>
    const T& observation::safe_cast() const {
        if (has_type<T>()) {
            return cast<T>();
        } else {
            throw std::bad_cast{};
        }
    }

    template<typename T>
    bool observation::has_type() const {
        return typeid(T) == obs_type;
    }

    template<typename T>
    const T& observation::cast() const {
        return *reinterpret_cast<const T*>(obs);
    }

} // namespace rtHMM

#endif //RTHMM_OBSERVATION_H
