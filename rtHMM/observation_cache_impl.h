#include <map>

namespace rtHMM {
    namespace internal {

        using namespace std;

        template<typename hmm_type>
        no_cache<hmm_type>::no_cache(const hmm_type& m) :
            model(m)
        {}

        template<typename hmm_type>
        void no_cache<hmm_type>::add_observation(const observation_type& obs)
        {
            current_obs = &obs;
        }

        template<typename hmm_type>
        double no_cache<hmm_type>::probability(size_t state_id) const
        {
            assert(state_id < model.num_states());
            return model.observation_distribution(state_id)->probability(*current_obs);
        }

        template<typename DT>
        cached_distribution<DT>::cached_distribution(const shared_ptr<DT>& dist) :
            distribution(dist)
        {}

        template<typename DT>
        void cached_distribution<DT>::set_observation(const value_type& obs)
        {
            computed = false;
            current_obs = &obs;
        }

        template<typename DT>
        double cached_distribution<DT>::probability() const
        {
            assert(current_obs != NULL);
            if (!computed) {
                prob = distribution->probability(*current_obs);
                computed = true;
            }

            return prob;
        }

        template<typename HT>
        tied_cache<HT>::tied_cache(const HT& model)
        {
            map<const distribution_type*, shared_ptr<cached_distribution<distribution_type>>> dist_groups;

            for (size_t i = 0; i < model.num_states(); ++i) {
                const auto& obs_dist = model.observation_distribution(i);

                auto it = dist_groups.find(obs_dist.get());

                if (it == dist_groups.end()) {
                    auto inserted = dist_groups.emplace(obs_dist.get(),
                                        make_shared<cached_distribution<distribution_type>>(obs_dist));

                    distributions.push_back(inserted.first->second);
                } else {
                    distributions.push_back(it->second);
                }
            }
        }

        template<typename HT>
        void tied_cache<HT>::add_observation(const observation_type& obs)
        {
            for (auto& dist : distributions) {
                dist->set_observation(obs);
            }
        }

        template<typename HT>
        double tied_cache<HT>::probability(size_t state_id) const
        {
            assert(state_id < distributions.size());
            return distributions[state_id]->probability();
        }

    }
} // namespace rtHMM::internal
