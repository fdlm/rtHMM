#include <iostream>
#include <vector>
#include <list>
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "hmm.h"

using namespace std;
using namespace rtHMM;

#include "hmm_example/hmm_matrices.h"

/* TODO:
 *
 * The following things still need test cases:
 *
 *   - if the predecessors are created correctly
 *   - if resetting a transition works correctly
 *   - if tied observation distributions are created correctly
 *
 */


TEST(hmm_constructor_tests, test_num_states)
{
    const size_t num_states = 3;
    hmm<size_t> my_hmm(num_states);
    ASSERT_EQ(my_hmm.num_states(), num_states);
}

template<typename T, typename HMM_T>
void check_prior(const T& prior, const HMM_T& my_hmm)
{
    size_t i = 0;
    for (auto p : prior) {
        ASSERT_DOUBLE_EQ(my_hmm.prior(i), p);
        ++i;
    }
}

TEST(hmm_constructor_tests, test_prior_vector)
{
    const vector<double> prior{PRIOR};

    hmm<size_t> my_hmm(prior);
    ASSERT_EQ(my_hmm.num_states(), prior.size());
    check_prior(prior, my_hmm);
}

TEST(hmm_constructor_tests, test_prior_list)
{
    const list<double> prior{PRIOR};

    hmm<size_t> my_hmm(prior);
    ASSERT_EQ(my_hmm.num_states(), prior.size());
    check_prior(prior, my_hmm);
}

class hmm_test : public ::testing::Test {
    protected:
        virtual void SetUp() {

            dense_vector prior(STATE_COUNT);
            prior << PRIOR;
            dense_matrix trans(STATE_COUNT, STATE_COUNT);
            trans << TRANSITION;
            dense_matrix obs(STATE_COUNT, ALPHABET_SIZE);
            obs << OBSERVATION;

            simple_hmm = new disc_hmm(discrete_hmm(prior, trans, obs));
        }

        virtual void TearDown() {
            delete simple_hmm;
        }

        disc_hmm* simple_hmm;
};


TEST_F(hmm_test, prior_probability_test)
{
    double correct_prior_probs[] = {PRIOR};

    for (size_t i = 0; i < simple_hmm->num_states(); ++i) {
        ASSERT_DOUBLE_EQ(correct_prior_probs[i], simple_hmm->prior(i));
    }
}


TEST_F(hmm_test, successors_test)
{
    double correct_trans_probs[] = {TRANSITION};

    for (size_t i = 0; i < simple_hmm->num_states(); ++i) {
        size_t k = 0;
        auto& successors = simple_hmm->successors(i);

        for (size_t j = 0; j < simple_hmm->num_states(); ++j) {
            double correct_p = correct_trans_probs[i * STATE_COUNT + j];
            if (correct_p > 0.0) {
                auto& succ = successors[k];
                ASSERT_EQ(succ.state_id, j);
                ASSERT_DOUBLE_EQ(succ.probability, correct_p);
                ++k;
            }
        }
    }
}


TEST_F(hmm_test, observation_probability_test)
{
    double correct_observation_probs[] = {OBSERVATION};
    size_t obs[] = ALPHABET;

    for (size_t state_id = 0; state_id < simple_hmm->num_states(); ++state_id) {
        for (size_t obs_id = 0; obs_id < ALPHABET_SIZE; ++obs_id) {
            auto& dist = simple_hmm->observation_distribution(state_id);
            double correct_p = correct_observation_probs[state_id * ALPHABET_SIZE + obs_id];
            ASSERT_DOUBLE_EQ(correct_p, dist->probability(obs[obs_id]));
        }
    }
}
