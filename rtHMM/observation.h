#ifndef RTHMM_OBSERVATION_H
#define RTHMM_OBSERVATION_H

#include <typeinfo>

namespace rtHMM {

    /*! \brief This is a type-erasure class that holds the reference to
     *         an HMM observation. The observation can be of any type, but
     *         must correspond to what the observation distributions of an
     *         HMM expect.
     *
     *  \sa hmm
     *  \sa filtering
     *  \sa decoding
     */
    class observation {
        public:
            /*! \brief Constructs the observation, erasing the type
             *  \tparam T type of observation
             *  \param[in] obs observation
             */
            template<typename T>
            observation(const T& obs);

            /*! \brief Casts the observation to a specific type, checking
             *         if the cast is correct (i.e. if the original observation
             *         was of the requested type
             *  \tparam T type to cast the observation to
             */
            template<typename T>
            const T& safe_cast() const;

            /*! \brief Checks if the observation is of a given type
             *  \tparam T Type to check
             */
            template<typename T>
            bool has_type() const;

            /*! \brief Casts the observation to a specific type, WITHOUT
             *         checking if the cast is valid.
             *  \tparam T type to cast the observation to
             */
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
