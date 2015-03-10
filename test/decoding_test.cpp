#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "decoding.h"
#include "hmm.h"

#include "hmm_example/hmm_matrices.h"
#include "hmm_example/hmm_sequence.h"

using namespace rtHMM;
using namespace std;

TEST(decoding_test, test_discrete_emission_algorithm)
{
    dense_vector prior{STATE_COUNT};
    dense_matrix trans{STATE_COUNT, STATE_COUNT};
    dense_matrix obs{STATE_COUNT, ALPHABET_SIZE};
    prior << PRIOR;
    trans << TRANSITION;
    obs << OBSERVATION;

    hmm model{prior, trans, obs};
    decoding dcd{model, 0.0, 2000};

    size_t obs_seq[] = TEST_SEQUENCE;
    double correct_vtrb[][STATE_COUNT] = VITERBI_RESULTS;

    for (size_t j = 0; j < STATE_COUNT; ++j) {
        double dec_val = dcd.viterbi_variables()[j];
        ASSERT_NEAR(correct_vtrb[0][j], dec_val, 0.00001);
    }

    for (size_t i = 0; i < TEST_SEQUENCE_LENGTH; ++i) {
        dcd.add_observation(obs_seq[i]);
        for (size_t j = 0; j < STATE_COUNT; ++j) {
            double dec_val = dcd.viterbi_variables()[j];
            ASSERT_NEAR(correct_vtrb[i + 1][j], dec_val, 0.00001);
        }
    }

    size_t correct_state_seq[] = DECODED_STATE_SEQUENCE;
    vector<size_t> seq = dcd.state_sequence();

    for (int i = 0; i < TEST_SEQUENCE_LENGTH - 1; ++i) {
        ASSERT_EQ(correct_state_seq[i], seq[i]);
    }

    ASSERT_NEAR(STATE_SEQUENCE_LOGPROB, dcd.state_sequence_log_probability(), 0.00001);
}
