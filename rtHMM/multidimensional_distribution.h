#ifndef RTHMM_MULTIDIMENSIONAL_DISTRIBUTION_H
#define RTHMM_MULTIDIMENSIONAL_DISTRIBUTION_H

#include <tuple>

#include "distribution.h"

namespace rtHMM {


    namespace internal {
        template<typename multidim_dist_type, size_t i> struct multidim_prob_comp;
    };

    using namespace internal;
    using namespace std;

    /*! /brief A multidimensional distribution assuming independent dimensions.
     *         When evaluating the probability mass or density, the values for
     *         individual dimensions must be given as tuple.
     *
     *  \tparam first_dist_type Type of the distribution of the first dimension
     *  \tparam second_dist_type Type of the distribution of the second dimension
     *  \tparam dist_types.. Types of the distributions of the remaining dimensions
     *  \sa distribution
     * */
    template<typename first_dist_type, typename second_dist_type, typename... dist_types>
    class multidimensional_distribution : public distribution_base<tuple<typename first_dist_type::value_type,
                                                                         typename second_dist_type::value_type,
                                                                         typename dist_types::value_type...>> {
        public:
            typedef tuple<typename first_dist_type::value_type,
                          typename second_dist_type::value_type,
                          typename dist_types::value_type...> value_type;

            /*! \brief number of dimensions */
            static constexpr size_t num_dims = sizeof...(dist_types) + 2;

            /*! \brief Constructs a multidimensional distribution given the
             *         distributions of the individual dimensions.
             *  \param[in] first_dist Distribution of the first dimension
             *  \param[in] second_dist Distribution of the second dimension
             *  \param[in] dists... Distributions of the remaining dimensions
             */
            multidimensional_distribution(const first_dist_type& first_dist,
                                          const second_dist_type& second_dist,
                                          const dist_types& ... dists);

            virtual ~multidimensional_distribution() {}

        private:
            tuple<first_dist_type, second_dist_type, dist_types...> distributions;

            double compute_probability(const value_type& value) const override;

            template<typename multidim_dist_type, size_t i> friend struct rtHMM::internal::multidim_prob_comp;
    };
} // namespace rtHMM


// include template implementations
#include "multidimensional_distribution_impl.h"

#endif //RTHMM_MULTIDIMENSIONAL_DISTRIBUTION_H
