#ifndef RTHMM_DISCRETE_DISTRIBUTION_H
#define RTHMM_DISCRETE_DISTRIBUTION_H

#include <vector>
#include <cassert>

#include "distribution.h"

namespace rtHMM {

    using namespace std;

    /*! \brief A discrete single-dimensional probability distribution. Values
     *         are non-negative integers (no 'fancy' mapping, you can use an
     *         enum if you need to)
     */
    class discrete_distribution : public distribution<size_t> {
        public:
            /*! \brief Constructs the distribution using a probability table.
             *         You have to normalise the probability table yourself so
             *         that it sums to one.
             *  \tparam T Type of the probability table. Anything iterable should do.
             *  \param[in] probability_table
             *      \parblock Table (or rather, array) of probability masses.
             *      Value-to-probability mapping is simply defined by the index
             *      in this table.
             *
             *  \sa distribution
             */
            template<typename T>
            discrete_distribution(const T& probability_table)
                : probabilities(begin(probability_table), end(probability_table))
            {
            }

            virtual ~discrete_distribution() {}

        private:
            vector<double> probabilities;
            double compute_probability(const size_t& value) const override
            {
                assert(value < probabilities.size());
                return probabilities[value];
            }
    };

} // namespace rtHMM

#endif //RTHMM_DISCRETE_DISTRIBUTION_H
