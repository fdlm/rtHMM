#include <iostream>
#include <functional>
#include <numeric>

namespace rtHMM {

    using namespace std;

    template<typename HT, bool OT>
    filtering<HT, OT>::filtering(const HT& hmm_model, double skip_prb, size_t max_past_steps) :
        model(hmm_model),
        state_count(hmm_model.num_states()),
        skip_prob(skip_prb),
        memory_size(max(max_past_steps + 1, 2ul)),
        seq_prob(0.0),
        obs_calc(hmm_model)
    {
        // initialise first forward variable with the prior
        alpha.emplace_back(state_count, 0.0);
        auto& alpha_cur = alpha.back();
        double added_prob = 0.0;

        for (size_t i = 0; i < state_count; ++i) {
            double p = model.prior(i);
            if (p > 0.0) {
                alpha_cur[i] = p;
                nonzero_elements.emplace_back(i);
                added_prob += p;
            }

            // if added_prob >= 1.0, all other priors will be zero and
            // therefore can be ommited
            if (added_prob >= 1.0) {
                break;
            }
        }

        nonzero_elements.reserve(state_count);
        new_nonzero_elements.reserve(state_count);
    }

    template<typename HT, bool OT>
    void filtering<HT, OT>::add_observation(const typename HT::observation_type& obs)
    {
        const vector<double>& alpha_prev = alpha.back();

        if (alpha.size() == memory_size) {
            // put first element to the end and use it
            auto& new_alpha = alpha.front();
            fill(new_alpha.begin(), new_alpha.end(), 0.);
            alpha.splice(alpha.end(), alpha, alpha.begin());
        } else {
            alpha.emplace_back(vector<double>(state_count, 0.));
        }

        vector<double>& alpha_cur = alpha.back();
        new_nonzero_elements.clear();

        for (size_t i : nonzero_elements) {
            for (const auto& succ : model.successors(i)) {
                if (alpha_cur[succ.state_id] == 0.0) {
                    new_nonzero_elements.emplace_back(succ.state_id);
                }
                alpha_cur[succ.state_id] += alpha_prev[i] * succ.probability;
            }
        }

        double norm_val = 0.;
        obs_calc.add_observation(obs);
        for (size_t i : new_nonzero_elements) {
            alpha_cur[i] *= obs_calc.probability(i);
            norm_val += alpha_cur[i];
        }

        seq_prob += log(norm_val);
        double scale_factor = 1.0 / norm_val;
        if (isfinite(scale_factor) == 0) {
            cerr << "ERROR: scale factor not finite: " << scale_factor << ", " << norm_val << '\n';
        }

        nonzero_elements.clear();
        for (size_t i : new_nonzero_elements) {
            alpha_cur[i] *= scale_factor;

            if (alpha_cur[i] > skip_prob) {
                nonzero_elements.emplace_back(i);
            }
        }
    }

    template<typename HT, bool OT>
    template<class container_type>
    void filtering<HT, OT>::add_observation_sequence(const container_type& seq)
    {
        for (const auto& obs : seq) {
            add_observation(obs);
        }
    }

    template<typename HT, bool OT>
    size_t filtering<HT, OT>::n_past_steps() const
    {
        return alpha.size();
    }

    template<typename HT, bool OT>
    size_t filtering<HT, OT>::max_past_steps() const
    {
        return memory_size - 1;
    }

    template<typename HT, bool OT>
    double filtering<HT, OT>::sequence_probability() const
    {
        return exp(seq_prob);
    }

    template<typename HT, bool OT>
    double filtering<HT, OT>::sequence_log_probability() const
    {
        return seq_prob;
    }

    template<typename HT, bool OT>
    const vector<double>& filtering<HT, OT>::distribution(size_t back_steps) const
    {
        assert(n_past_steps() > back_steps);
        auto it = alpha.rbegin();
        advance(it, back_steps);
        return *it;
    }


} // namespace rtHMM
