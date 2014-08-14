#ifndef RTHMM_DISTRIBUTION_H
#define RTHMM_DISTRIBUTION_H

namespace rtHMM {

    /*! \brief This is the base class of all distributions used within
     *         the rtHMM framework. If you want to create you own distribution,
     *         you have to derive from it.
     *
     *  \tparam T Type of the value the distribution can be evaluated for
     */
    template<typename T>
    class distribution {
        public:
            typedef T value_type;

            distribution() {};
            virtual ~distribution() {};

            /*! \brief Evaluates the distribution (density or mass) for a
             *         value.
             *  \param[in] value value for which the distribution is evaluated
             *  \returns Probability density or mass (depending on the type of distribution) for the passed value.
             */
            double probability(const T& value) const {
                return compute_probability(value);
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
