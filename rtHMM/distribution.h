#ifndef RTHMM_DISTRIBUTION_H
#define RTHMM_DISTRIBUTION_H

#include "observation.h"


namespace rtHMM {

    /*! \brief This is the interface for probability distributions within the rtHMM framework.
     *         If you want to implement your own distribution, you have to derive at least
     *         from this class. Better yet, derive from distribution_base to get automatic
     *         type conversion and to be able to use your distribution within a
     *         mixture_distribution and multidimensional_distribution.
     *
     *  \sa distribution_base
     *  \sa mixture_distribution
     *  \sa multidimensional_distribution
     */
    class distribution {
        public:
            distribution() {};
            virtual ~distribution() {};

            /*! \brief Evaluates the distribution (density or mass) for a value.
             *  \param[in] value value for which the distribution is evaluated
             *  \returns Probability density or mass (depending on the type of distribution) for the passed value.
             */
            virtual double probability(const observation& value) const = 0;
    };


    /*! \brief This is the base class used for all distributions in the rtHMM framework.
     *         Derive from this class if you want to implement your own distribution.
     *  \tparam T  Value type for which the distribution is defined
     */
    template<typename T>
    class distribution_base : public distribution {
        public:
            using value_type = T;

            double probability(const observation& value) const override {
                return compute_probability(value.safe_cast<T>());
            }

        private:
            /*! \brief This function should compute the probability mass/density
             *         for a given value. Each distribution derived from this
             *         base class has to implement it.
             *  \param[in] value value for which the probability mass/density is evaluated
             *  \returns Probability mass/density for the value
             */
            virtual double compute_probability(const T& value) const = 0;
    };

} // namespace rtHMM

#endif //RTHMM_DISTRIBUTION_H
