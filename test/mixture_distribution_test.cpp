#include "gtest/gtest.h"
#include "mixture_distribution.h"
#include "gaussian_distribution.h"

using namespace rtHMM;

TEST(mixture_distribution_test, test_dist)
{
    const double weights[] = { 1.0, 2.0 };
    const double means[] = { 5.0, 10.0 };
    const double sigmas[] = { 16.0, 10.0 };
    const double testval = 7.2;
    const double sum_weights = 3.0;

    gaussian_distribution<1> g1(means[0], sigmas[0]);
    gaussian_distribution<1> g2(means[1], sigmas[1]);

    mixture_distribution<gaussian_distribution<1>, gaussian_distribution<1>> gmm({weights[0], weights[1]}, g1, g2);

    ASSERT_DOUBLE_EQ(weights[0] / sum_weights * g1.probability(testval) + weights[1] / sum_weights * g2.probability(testval),
                     gmm.probability(testval));
}
