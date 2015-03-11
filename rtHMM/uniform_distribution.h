#ifndef RTHMM_UNIFORM_DISTRIBUTION_H
#define RTHMM_UNIFORM_DISTRIBUTION_H

#include "distribution.h"

namespace rtHMM {

    /*! \brief A single-dimensional continuous uniform distribution
     *  \sa distribution
     */
    class uniform_distribution : public distribution_base<double> {
        public:
            /*! \brief Constructs the uniform distribution within a range.
             *         Outside of this range, the density is 0.
             *
             *  \param[in] _lower_bound Below this value, the density is 0
             *  \param[in] _upper_bound Above this value, the density is 0
             */
            uniform_distribution(double _lower_bound, double _upper_bound) :
                lower_bound(_lower_bound), upper_bound(_upper_bound),
                pdf_value(1.0 / (_upper_bound - lower_bound))
            {
                assert(upper_bound > lower_bound);
            }

        private:
            const double lower_bound;
            const double upper_bound;
            const double pdf_value;

            double compute_probability(const double& value) const override
            {
                return (value >= lower_bound && value <= upper_bound) ? pdf_value : 0.0;
            }
    };

} // namespace rtHMM

#endif // RTHMM_UNIFORM_DISTRIBUTION_H
