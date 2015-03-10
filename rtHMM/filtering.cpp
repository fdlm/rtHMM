#include "filtering.h"

#include <iostream>
#include <functional>
#include <numeric>

namespace rtHMM {

    using namespace std;
    using namespace internal;

    filtering::filtering(const hmm& hmm_model, double skip_prb, size_t max_past_steps, bool optimise_tied) :
        model(hmm_model),
        state_count(hmm_model.num_states()),
        skip_prob(skip_prb),
        memory_size(max(max_past_steps + 1, 2ul)),
        seq_prob(0.0)
    {
        // initialise first forward variable with the prior
        alpha.emplace_back(state_count, 0.0);
        auto& alpha_cur = alpha.back();
        double added_prob = 0.0;

        if (optimise_tied) {
            obs_prob_calc = make_unique<tied_cache>(hmm_model);
        } else{
            obs_prob_calc = make_unique<no_cache>(hmm_model);
        }

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

    void filtering::add_observation(const observation& obs)
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
        obs_prob_calc->add_observation(obs);
        for (size_t i : new_nonzero_elements) {
            alpha_cur[i] *= obs_prob_calc->probability(i);
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

    size_t filtering::n_past_steps() const
    {
        return alpha.size();
    }

    size_t filtering::max_past_steps() const
    {
        return memory_size - 1;
    }

    double filtering::sequence_probability() const
    {
        return exp(seq_prob);
    }

    double filtering::sequence_log_probability() const
    {
        return seq_prob;
    }

    const vector<double>& filtering::distribution(size_t back_steps) const
    {
        assert(n_past_steps() > back_steps);
        auto it = alpha.rbegin();
        advance(it, back_steps);
        return *it;
    }

} // namespace rtHMM
