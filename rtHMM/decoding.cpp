#include "decoding.h"

#include <iostream>

namespace rtHMM {

    using namespace std;
    using namespace internal;

    decoding::decoding(const hmm& hmm_model, double skip_prb, size_t max_lag, bool optimise_tied) :
        model(hmm_model),
        skip_prob(skip_prb),
        memory_size(max(max_lag, 1ul)),
        state_count(hmm_model.num_states()),
        viterbi_cur(&viterbi.back()),
        viterbi_prev(&viterbi.front()),
        total_scale_correction(0.0)
    {
        viterbi_cur->resize(state_count);
        viterbi_prev->resize(state_count);
        auto& vit_c = *viterbi_cur;
        double added_prob = 0.0;
        double max_prob = -1.0;

        if (optimise_tied) {
            obs_prob_calc = make_unique<tied_cache>(hmm_model);
        } else {
            obs_prob_calc = make_unique<no_cache>(hmm_model);
        }

        for (size_t i = 0; i < state_count; ++i) {
            double p = model.prior(i);

            if (p > 0.0) {
                vit_c[i] = p;
                nonzero_elements.emplace_back(i);
                added_prob += p;
            }

            if (p > max_prob) {
                most_probable_end = i;
            }

            // if added_prob >= 1.0, all other priors will be zero and
            // therefore can be ommited
            if (added_prob >= 1.0) {
                break;
            }
        }
    }

    void decoding::add_observation(const observation& obs)
    {
        auto tmp = viterbi_prev;
        viterbi_prev = viterbi_cur;
        viterbi_cur = tmp;

        auto& v_cur = *viterbi_cur;
        auto& v_prev = *viterbi_prev;
        fill(v_cur.begin(), v_cur.end(), 0.);

        if (backtracking_pointers.size() == memory_size) {
            // put first element to the end and use it
            backtracking_pointers.splice(backtracking_pointers.end(),
                                         backtracking_pointers,
                                         backtracking_pointers.begin());

        } else {
            backtracking_pointers.emplace_back(state_count, 0);
        }

        auto& backtracking_ptr = backtracking_pointers.back();

        for (size_t i : nonzero_elements) {
            for (const auto& succ : model.successors(i)) {
                double reach_prob = v_prev[i] * succ.probability;

                if (reach_prob > v_cur[succ.state_id]) {
                    if (v_cur[succ.state_id] == 0.0) {
                        new_nonzero_elements.emplace_back(succ.state_id);
                    }

                    v_cur[succ.state_id] = reach_prob;
                    backtracking_ptr[succ.state_id] = i;
                }
            }
        }

        nonzero_elements.clear();

        double probability_sum = 0.0;

        obs_prob_calc->add_observation(obs);
        for (size_t i : new_nonzero_elements) {
            v_cur[i] *= obs_prob_calc->probability(i);
            probability_sum += v_cur[i];
        }

        total_scale_correction += log(probability_sum);
        double scale_factor = 1.0 / probability_sum;
        if (isfinite(scale_factor) == 0) {
            cerr << "ERROR: scale factor not finite: " << scale_factor << ", " << probability_sum << '\n';
        }

        double max_prob = 0.0;
        for (size_t i : new_nonzero_elements) {
            v_cur[i] *= scale_factor;

            if (v_cur[i] > skip_prob) {
                nonzero_elements.emplace_back(i);

                if (v_cur[i] > max_prob) {
                    max_prob = v_cur[i];
                    most_probable_end = i;
                }
            }
        }
        new_nonzero_elements.clear();
    }

    const vector<double>& decoding::viterbi_variables() const
    {
        return *viterbi_cur;
    }

    vector<size_t> decoding::state_sequence() const
    {
        return state_sequence(backtracking_pointers.size());
    }

    vector<size_t> decoding::state_sequence(size_t back_steps) const
    {
        assert(back_steps <= backtracking_pointers.size());

        vector<size_t> sequence(back_steps + 1);
        size_t max_id = most_probable_end;

        sequence[back_steps] = max_id;

        auto bt_pointers_it = backtracking_pointers.rbegin();
        while (back_steps-- > 0) {
            max_id = (*bt_pointers_it).at(max_id);
            bt_pointers_it++;
            sequence[back_steps] = max_id;
        }

        return sequence;
    }

    double decoding::state_sequence_log_probability() const
    {
        return log(viterbi_cur->at(most_probable_end)) + total_scale_correction;
    }

    double decoding::state_sequence_probability() const
    {
        return exp(state_sequence_log_probability());
    }

    size_t decoding::n_past_steps() const
    {
        return backtracking_pointers.size();
    }

    size_t decoding::max_past_steps() const
    {
        return memory_size;
    }

} // namespace rtHMM
