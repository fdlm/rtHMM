#include "hmm.h"

namespace rtHMM {

    namespace internal {

        void insert_or_update(hmm::link&& lnk, vector<hmm::link>& links)
        {
            auto elem = find(begin(links), end(links), lnk);
            if (elem == links.end()) {
                links.emplace_back(lnk);
            } else {
                elem->probability = lnk.probability;
            }
        }

        void init_prior(const dense_vector& prior, vector<double>& model_prior)
        {
            assert(prior.rows() > 0);
            size_t num_rows = abs(prior.rows());

            model_prior.resize(num_rows);
            for (size_t i = 0; i < num_rows; ++i) {
                model_prior[i] = prior(i);
            }
        }

        void init_prior(const sparse_vector& prior, vector<double>& model_prior)
        {
            assert(prior.rows() > 0);
            size_t num_rows = abs(prior.rows());

            model_prior.resize(num_rows);
            fill(begin(model_prior), end(model_prior), 0.);

            for (sparse_vector::InnerIterator it(prior); it; ++it) {
                model_prior[it.index()] = it.value();
            }
        }

        void init_transition(const dense_matrix& trans, hmm& model)
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

        void init_transition(const sparse_matrix& trans, hmm& model)
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

        void set_discrete_observations(const dense_matrix& obs, hmm& model)
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
                model.set_observation_distribution(i, dist);
            }
        }

    } // namespace rtHMM::internal

    void hmm::set_prior(size_t state_id, double probability)
    {
        assert(state_id < num_states());

        prior_probs[state_id] = probability;
    }

    void hmm::set_transition(size_t from_state_id, size_t to_state_id, double transition_prob)
    {
        assert(from_state_id < num_states());
        assert(to_state_id < num_states());

        internal::insert_or_update({to_state_id, transition_prob}, successor_links[from_state_id]);
        internal::insert_or_update({from_state_id, transition_prob}, predecessor_links[to_state_id]);
    }

    void hmm::set_observation_distribution(size_t state_id, shared_ptr<distribution> dist)
    {
        assert(state_id < num_states());
        observation_dists[state_id] = dist;
    }

} // namespace rtHMM
