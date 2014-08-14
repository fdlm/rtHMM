namespace rtHMM {

    using namespace std;

    template<typename FDT, typename SDT, typename... DT>
    multidimensional_distribution<FDT, SDT, DT...>::multidimensional_distribution(const FDT& first_dist, const SDT& second_dist, const DT& ... dists) :
        distributions(first_dist, second_dist, dists...)
    {}

    namespace internal {

        template<typename multidim_dist_type, size_t i>
        struct multidim_prob_comp {
            static_assert(i < multidim_dist_type::num_dims,
                          "Index i must be smaller than the number of dimensions in the multidimesional distribution");

            static double compute(const multidim_dist_type& multidim_dist, const typename multidim_dist_type::value_type& val) 
            {
                return get<i>(multidim_dist.distributions).probability(get<i>(val)) *
                       multidim_prob_comp<multidim_dist_type, i - 1>::compute(multidim_dist, val);
            }
        };

        template<typename multidim_dist_type>
        struct multidim_prob_comp<multidim_dist_type, 0> {
            static double compute(const multidim_dist_type& multidim_dist, const typename multidim_dist_type::value_type& val) 
            {
                return get<0>(multidim_dist.distributions).probability(get<0>(val));
            }
        };

    } // namespace internal

    template<typename FDT, typename SDT, typename... DT>
    double multidimensional_distribution<FDT, SDT, DT...>::compute_probability(const typename multidimensional_distribution<FDT, SDT, DT...>::value_type & value) const
    {
        return internal::multidim_prob_comp<multidimensional_distribution<FDT, SDT, DT...>, num_dims - 1>::compute(*this, value);
    }

} // namespace rtHMM
