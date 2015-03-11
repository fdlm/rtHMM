#include "observation_cache.h"

#include <map>
#include "hmm.h"

namespace rtHMM {

    namespace internal {

        using namespace std;

        no_cache::no_cache(const hmm& m) :
            model(m)
        {}

        void no_cache::add_observation(const observation& obs)
        {
            current_obs = &obs;
        }

        double no_cache::probability(size_t state_id) const
        {
            assert(state_id < model.num_states());
            return model.observation_distribution(state_id)->probability(*current_obs);
        }

        cached_distribution::cached_distribution(const shared_ptr<distribution>& dist) :
            dist(dist)
        {}

        void cached_distribution::set_observation(const observation& obs)
        {
            computed = false;
            current_obs = &obs;
        }

        double cached_distribution::probability() const
        {
            assert(current_obs != NULL);
            if (!computed) {
                prob = dist->probability(*current_obs);
                computed = true;
            }

            return prob;
        }

        tied_cache::tied_cache(const hmm& model)
        {
            map<const distribution*, shared_ptr<cached_distribution>> dist_groups;

            for (size_t i = 0; i < model.num_states(); ++i) {
                const auto& obs_dist = model.observation_distribution(i);

                auto it = dist_groups.find(obs_dist.get());

                if (it == dist_groups.end()) {
                    auto inserted = dist_groups.emplace(obs_dist.get(), make_shared<cached_distribution>(obs_dist));
                    dists.push_back(inserted.first->second);
                } else {
                    dists.push_back(it->second);
                }
            }
        }

        void tied_cache::add_observation(const observation& obs)
        {
            for (auto& dist : dists) {
                dist->set_observation(obs);
            }
        }

        double tied_cache::probability(size_t state_id) const
        {
            assert(state_id < dists.size());
            return dists[state_id]->probability();
        }

    } // namespace rtHMM::internal

} // namespace rtHMM
