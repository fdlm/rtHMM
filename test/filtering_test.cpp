#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "gtest/gtest.h"
#include "filtering.h"
#include "hmm.h"
#include "discrete_distribution.h"

#include "hmm_example/hmm_matrices.h"
#include "hmm_example/hmm_sequence.h"

using namespace rtHMM;
using namespace std;


/* TODO:
 *
 * The following things still need test cases:
 *
 *   - Forward algorithm with continuous distribution (like gaussian).
 *     Actually, this shouldn't be a problem if the distribution works properly
 *
 */

TEST(filtering_test, test_discrete_emission)
{
    dense_vector prior(STATE_COUNT);
    dense_matrix trans(STATE_COUNT, STATE_COUNT);
    dense_matrix obs(STATE_COUNT, ALPHABET_SIZE);
    prior << PRIOR;
    trans << TRANSITION;
    obs << OBSERVATION;

    disc_hmm model = discrete_hmm(prior, trans, obs);

    filtering<disc_hmm> fltr {model};
    size_t obs_seq[] = TEST_SEQUENCE;

    double correct_fwd_vars[][STATE_COUNT] = FILTERING_RESULTS;

    for (size_t j = 0; j < STATE_COUNT; ++j) {
        double fwd_val = fltr.distribution()[j];
        ASSERT_NEAR(correct_fwd_vars[0][j], fwd_val, 0.00001);
    }

    for (size_t i = 0; i < TEST_SEQUENCE_LENGTH; ++i) {
        fltr.add_observation(obs_seq[i]);
        for (size_t j = 0; j < STATE_COUNT; ++j) {
            double fwd_val = fltr.distribution()[j];
            ASSERT_NEAR(correct_fwd_vars[i + 1][j], fwd_val, 0.00001);
        }
    }

    ASSERT_NEAR(exp(SEQUENCE_LOGPROB), fltr.sequence_probability(), 0.00001);
    ASSERT_NEAR(SEQUENCE_LOGPROB, fltr.sequence_log_probability(), 0.00001);
}
