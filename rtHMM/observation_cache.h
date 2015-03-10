#ifndef RTHMM_OBSERVATION_CACHE_H
#define RTHMM_OBSERVATION_CACHE_H

#include <memory>
#include <vector>

#include "distribution.h"

namespace rtHMM {

    class hmm;

    namespace internal {

        using namespace std;


        class observation_cache {
            public:
                virtual void add_observation(const observation& obs) = 0;
                virtual double probability(size_t state_id) const = 0;
        };


        class no_cache : public observation_cache {
            public:
                no_cache(const hmm& model);

                void add_observation(const observation& obs) override;
                double probability(size_t state_id) const override;

            private:
                const hmm& model;
                const observation* current_obs = nullptr;
        };


        class cached_distribution {
            public:
                cached_distribution(const shared_ptr<distribution>& dist);

                void set_observation(const observation& obs);
                double probability() const;

            private:
                mutable double prob = 0.0;
                mutable bool computed = false;
                const observation* current_obs = nullptr;
                const shared_ptr<distribution> dist;
        };

        class tied_cache : public observation_cache {
            public:
                tied_cache(const hmm& model);

                void add_observation(const observation& obs) override;
                double probability(size_t state_id) const override;

            private:
                vector<shared_ptr<cached_distribution>> dists;
        };

    } // namespace rtHMM::internal

} // namespace rtHMM

#endif // RTHMM_OBSERVATION_CACHE_H
