#include "gtest/gtest.h"
#include "discrete_distribution.h"
#include <vector>

using namespace std;

#define DISCRETE_1D_PROBS { 0.1, 0.5, 0.3, 0.1, 0.0 }

TEST(discrete_distribution_test, test_1d_dist)
{
    double corr_probs[] = DISCRETE_1D_PROBS;
    size_t alphabet_size = distance(begin(corr_probs), end(corr_probs));

    vector<double> ps(DISCRETE_1D_PROBS);

    rtHMM::discrete_distribution my_1d_dist(ps);

    for (size_t i = 0; i < alphabet_size; ++i) {
        ASSERT_DOUBLE_EQ(corr_probs[i], my_1d_dist.probability(i));
    }
}
