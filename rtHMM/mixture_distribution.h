#ifndef RTHMM_MIXTURE_DISTRIBUTION_H
#define RTHMM_MIXTURE_DISTRIBUTION_H

#include <list>
#include <utility>
#include <memory>
#include <tuple>
#include <array>

#include "distribution.h"
#include "utils.h"


namespace rtHMM {

    using namespace std;
    using namespace internal;

    namespace internal {
        template<typename mixed_dist_type, size_t i> struct mixed_prob_comp;
    };

    /*! \brief A mixture of multiple distributions over the same dimensions.
     *         The mixture components do not have to be of the same type.
     *  \tparam first_dist_type Type of the first mixture component
     *  \tparam second_dist_type Type of the second mixture component
     *  \tparam dist_types... Types of the remaining mixture components
     *  \sa distribution
     */
    template<typename first_dist_type, typename second_dist_type, typename... dist_types>
    class mixture_distribution : public distribution_base<typename first_dist_type::value_type> {

        static_assert(are_same<typename first_dist_type::value_type,
                               typename second_dist_type::value_type,
                               typename dist_types::value_type...>::value == 1,
                      "Distributions must be defined over same data type");

        public:
            /*! \brief number of mixture components */
            static constexpr size_t num_dists = sizeof...(dist_types) + 2;

            /*! \brief Constructs a mixture distribution given the weights of
             *         the components and the respective distributions
             *  \param[in] weights Array containing the weights of the components
             *  \param[in] first_dist First component distribution
             *  \param[in] second_dist Second component distribution
             *  \param[in] dists... Remaining component distributions
             */
            mixture_distribution(const array<double, num_dists> weights,
                                 const first_dist_type& first_dist, const second_dist_type& second_dist,
                                 const dist_types&... dists);

            virtual ~mixture_distribution() {}

        private:
            tuple<first_dist_type, second_dist_type, dist_types...> distributions;
            array<double, num_dists> weights;

            double compute_probability(const typename first_dist_type::value_type& value) const override;

            template<typename mixed_dist_type, size_t i> friend struct rtHMM::internal::mixed_prob_comp;
    };

} // namespace rtHMM

#include "mixture_distribution_impl.h"

#endif // RTHMM_MIXTURE_DISTRIBUTION_H
