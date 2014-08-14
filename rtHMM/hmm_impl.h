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

        template<>
        inline void init_prior(const dense_vector& prior, vector<double>& model_prior)
        {
            assert(prior.rows() > 0);
            size_t num_rows = abs(prior.rows());

            model_prior.resize(num_rows);
            for (size_t i = 0; i < num_rows; ++i) {
                model_prior[i] = prior(i);
            }
        }

        template<>
        inline void init_prior(const sparse_vector& prior, vector<double>& model_prior)
        {
            assert(prior.rows() > 0);
            size_t num_rows = abs(prior.rows());

            model_prior.resize(num_rows);
            fill(begin(model_prior), end(model_prior), 0.);

            for (sparse_vector::InnerIterator it(prior); it; ++it) {
                model_prior[it.index()] = it.value();
            }
        }

        template<typename T, typename OT>
        inline void init_transition(const T& transition, hmm<OT>& model)
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

        template<typename OT>
        inline void init_transition(const dense_matrix& trans, hmm<OT>& model)
        {
            assert(trans.rows() > 0);
            assert(trans.rows() == trans.cols());
            assert(static_cast<size_t>(trans.rows()) == model.num_states());

            for (size_t i = 0; i < model.num_states(); ++i) {
                for (size_t j = 0; j < model.num_states(); ++j) {
                    if (trans(i, j) > 0.f) {
                        model.set_transition(i, j, trans(i, j));
                    }
                }
            }
        }

        template<typename OT>
        inline void init_transition(const sparse_matrix& trans, hmm<OT>& model)
        {
            assert(trans.rows() > 0);
            assert(trans.rows() == trans.cols());
            assert(static_cast<size_t>(trans.rows()) == model.num_states());

            for (int k = 0; k < trans.outerSize(); ++k) {
                for (sparse_matrix::InnerIterator it(trans, k); it; ++it) {
                    model.set_transition(it.row(), it.col(), it.value());
                }
            }
        }

        template<typename OT>
        inline void insert_or_update(typename hmm<OT>::link&& lnk, vector<typename hmm<OT>::link>& links)
        {
            auto elem = find(begin(links), end(links), lnk);
            if (elem == links.end()) {
                links.emplace_back(lnk);
            } else {
                elem->probability = lnk.probability;
            }
        }

    } // namespace internal

    template<typename OT>
    hmm<OT>::hmm(size_t num_states)
    {
        prior_probs.resize(num_states);
        successor_links.resize(num_states);
        predecessor_links.resize(num_states);
        observation_dists.resize(num_states);

        assert(prior_probs.size() == successor_links.size());
        assert(prior_probs.size() == predecessor_links.size());
        assert(prior_probs.size() == observation_dists.size());
    }

    template<typename OT>
    template<typename PT>
    hmm<OT>::hmm(const PT& prior)
    {
        internal::init_prior(prior, prior_probs);

        successor_links.resize(num_states());
        predecessor_links.resize(num_states());
        observation_dists.resize(num_states());

        assert(prior_probs.size() == successor_links.size());
        assert(prior_probs.size() == predecessor_links.size());
        assert(prior_probs.size() == observation_dists.size());
    }

    template<typename OT>
    template<typename PT, typename TT>
    hmm<OT>::hmm(const PT& prior, const TT& transition) :
        hmm(prior)
    {
        internal::init_transition(transition, *this);
    }


    template<typename OT>
    void hmm<OT>::set_transition(size_t from_state_id, size_t to_state_id, double transition_prob)
    {
        assert(from_state_id < num_states());
        assert(to_state_id < num_states());

        internal::insert_or_update<OT>({to_state_id, transition_prob}, successor_links[from_state_id]);
        internal::insert_or_update<OT>({from_state_id, transition_prob}, predecessor_links[to_state_id]);
    }

    template<typename OT>
    void hmm<OT>::set_observation_distribution(size_t state_id,
            shared_ptr<hmm<OT>::distribution_type> dist)
    {
        assert(state_id < num_states());
        observation_dists[state_id] = dist;
    }

    template<typename OT>
    template<typename T>
    void hmm<OT>::set_tied_observation_distribution(const T& state_ids,
            shared_ptr<hmm<OT>::distribution_type> dist)
    {
        for (size_t state_id : state_ids) {
            set_observation_distribution(state_id, dist);
        }
    }


    namespace internal {

        template<typename T>
        inline void set_discrete_observations(disc_hmm& model, const T& obs)
        {
            assert(obs.size() > 0);
            assert(obs.size() == model.num_states());
            size_t num_symbols = begin(obs)->size();

            size_t i = 0;
            for (const auto& probs : obs) {
                assert(probs.size() == num_symbols);
                auto dist = make_shared<discrete_distribution>(probs);
                model.set_observation_distribution(i, static_pointer_cast<distribution<size_t>>(dist));
                ++i;
            }
        }

        template<>
        inline void set_discrete_observations(disc_hmm& model, const dense_matrix& obs)
        {
            assert(obs.rows() > 0);
            assert(static_cast<size_t>(obs.rows()) == model.num_states());
            size_t num_symbols = obs.cols();

            for (size_t i = 0; i < model.num_states(); ++i) {
                vector<double> probs(num_symbols);
                for (size_t j = 0; j < num_symbols; ++j) {
                    probs[j] = obs(i, j);
                }
                auto dist = make_shared<discrete_distribution>(probs);
                model.set_observation_distribution(i, static_pointer_cast<distribution<size_t>>(dist));
            }
        }

    } // namespace internal

    template<typename T1, typename T2, typename T3>
    disc_hmm discrete_hmm(const T1& prior, const T2& transition, const T3& obs)
    {
        disc_hmm model{prior, transition};
        internal::set_discrete_observations(model, obs);

        return model;
    }

} // namespace rtHMM
