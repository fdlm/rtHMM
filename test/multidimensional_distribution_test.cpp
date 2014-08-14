#include <vector>
#include <tuple>
#include "gtest/gtest.h"
#include "multidimensional_distribution.h"
#include "gaussian_distribution.h"
#include "discrete_distribution.h"


TEST(multidimensional_distribution_test, test_dist)
{
    std::vector<double> pt({0.2, 0.3, 0.5});
    rtHMM::gaussian_distribution<1> g(5.0, 3.0);
    rtHMM::discrete_distribution d(pt);

    double g_val = 3.4;
    size_t d_val = 1;

    rtHMM::multidimensional_distribution<rtHMM::gaussian_distribution<1>, rtHMM::discrete_distribution> md(g, d);

    ASSERT_DOUBLE_EQ(g.probability(g_val) * d.probability(d_val), md.probability(std::make_tuple(g_val, d_val)));
}
