#include <cassert>


namespace rtHMM {

    using namespace std;

    namespace internal {

        template<typename T>
        inline void init_prior(const T& prior, vector<double>& model_prior)
        {
            // assuming T is a container containing something that's
            // castable to doubles
            model_prior.clear();
            copy(begin(prior), end(prior), back_inserter(model_prior));
        }

        template<typename T>
        inline void init_transition(const T& transition, hmm& model)
        {
            int i = 0;
            for (const auto& row : transition) {
                int j = 0;
                for (auto p : row) {
                    model.set_transition(i, j, p);
                    ++j;
                }
                ++i;
            }
        }

        template<typename T>
        inline void set_discrete_observations(const T& obs, hmm& model)
        {
            assert(obs.size() > 0);
            assert(obs.size() == model.num_states());
            size_t num_symbols = begin(obs)->size();

            size_t i = 0;
            for (const auto& probs : obs) {
                assert(probs.size() == num_symbols);
                auto dist = make_shared<discrete_distribution>(probs);
                model.set_observation_distribution(i, dist);
                ++i;
            }
        }

        void init_prior(const dense_vector& prior, vector<double>& model_prior);
        void init_prior(const sparse_vector& prior, vector<double>& model_prior);
        void init_prior(const sparse_vector& prior, vector<double>& model_prior);
        void init_transition(const dense_matrix& trans, hmm& model);
        void init_transition(const sparse_matrix& trans, hmm& model);
        void set_discrete_observations(const dense_matrix& obs, hmm& model);

    } // namespace internal

    template<typename PT>
    hmm::hmm(const PT& prior)
    {
        internal::init_prior(prior, prior_probs);

        successor_links.resize(num_states());
        predecessor_links.resize(num_states());
        observation_dists.resize(num_states());

        assert(prior_probs.size() == successor_links.size());
        assert(prior_probs.size() == predecessor_links.size());
        assert(prior_probs.size() == observation_dists.size());
    }

    template<typename PT, typename TT>
    hmm::hmm(const PT& prior, const TT& transition) :
        hmm(prior)
    {
        internal::init_transition(transition, *this);
    }

    template<typename T1, typename T2, typename T3>
    hmm::hmm(const T1& prior, const T2& transition, const T3& obs) :
        hmm(prior, transition)
    {
        internal::set_discrete_observations(obs, *this);
    }

    template<typename T>
    void hmm::set_tied_observation_distribution(const T& state_ids, shared_ptr<distribution> dist)
    {
        for (size_t state_id : state_ids) {
            set_observation_distribution(state_id, dist);
        }
    }


} // namespace rtHMM
