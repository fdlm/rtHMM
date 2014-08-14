#ifndef RTHMM_OBSERVATION_CACHE_H
#define RTHMM_OBSERVATION_CACHE_H

namespace rtHMM {
    namespace internal {

        using namespace std;

        template<typename hmm_type>
        class no_cache {
            public:
                typedef typename hmm_type::observation_type observation_type;

                no_cache(const hmm_type& model);

                void add_observation(const observation_type& obs);
                double probability(size_t state_id) const;

            private:
                const hmm_type& model;
                const observation_type* current_obs = NULL;
        };

        template<typename distribution_type>
        class cached_distribution {
            public:
                typedef typename distribution_type::value_type value_type;

                cached_distribution(const shared_ptr<distribution_type>& dist);

                void set_observation(const value_type& obs);
                double probability() const;

            private:
                mutable double prob = 0.0;
                mutable bool computed = false;
                const value_type* current_obs = NULL;
                const shared_ptr<distribution_type> distribution;
        };

        template<typename hmm_type>
        class tied_cache {
            public:
                typedef typename hmm_type::distribution_type distribution_type;
                typedef typename hmm_type::observation_type observation_type;

                tied_cache(const hmm_type& model);
                void add_observation(const observation_type& obs);
                double probability(size_t state_id) const;

            private:
                vector<shared_ptr<cached_distribution<distribution_type>>> distributions;
        };

        template<bool tie_observations_probabilities, typename hmm_type>
        struct cache_type;

        template<typename hmm_type>
        struct cache_type<true, hmm_type> {
            typedef tied_cache<hmm_type> type;
        };

        template<typename hmm_type>
        struct cache_type<false, hmm_type> {
            typedef no_cache<hmm_type> type;
        };

    } // namespace internal
} // namespace rtHMM

// implementation of template methods
#include "observation_cache_impl.h"

#endif // RTHMM_OBSERVATION_CACHE_H
