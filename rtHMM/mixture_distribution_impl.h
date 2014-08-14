#include <numeric>

namespace rtHMM {

    using namespace std;

    template<typename FDT, typename SDT, typename... DT>
    mixture_distribution<FDT, SDT, DT...>::mixture_distribution(array<double, num_dists> w, const FDT& first_dist,
            const SDT& second_dist, const DT& ... dists) :
        distributions(first_dist, second_dist, dists...),
        weights(w)
    {
        double weight_sum = accumulate(begin(weights), end(weights), 0.0);
        for (auto& weight : weights) {
            weight /= weight_sum;
        }
    }

    namespace internal {

        template<typename mixed_dist_type, size_t i>
        struct mixed_prob_comp {
            static_assert(i < mixed_dist_type::num_dists, "Index i must be smaller than the number of distributions in the mixture");

            static double compute(const mixed_dist_type& mixed_dist, const typename mixed_dist_type::value_type& val) {
                return get<i>(mixed_dist.weights) * get<i>(mixed_dist.distributions).probability(val) +
                       mixed_prob_comp<mixed_dist_type, i - 1>::compute(mixed_dist, val);
            }
        };

        template<typename mixed_dist_type>
        struct mixed_prob_comp<mixed_dist_type, 0> {
            static double compute(const mixed_dist_type& mixed_dist, const typename mixed_dist_type::value_type& val) {
                return get<0>(mixed_dist.weights) * get<0>(mixed_dist.distributions).probability(val);
            }
        };

    } // namespace internal

    template<typename FDT, typename SDT, typename... DT>
    double mixture_distribution<FDT, SDT, DT...>::compute_probability(const typename FDT::value_type& value) const
    {
        return internal::mixed_prob_comp<mixture_distribution<FDT, SDT, DT...>, num_dists - 1>::compute(*this, value);
    }

} // namespace rtHMM
