#include "gtest/gtest.h"
#include "uniform_distribution.h"

using namespace rtHMM;
using namespace std;

TEST(uniform_distribution_test, test_pdf)
{
    uniform_distribution dist(2.0, 4.0);

    ASSERT_DOUBLE_EQ(dist.probability(1.9), 0.0);
    ASSERT_DOUBLE_EQ(dist.probability(2.0), 0.5);
    ASSERT_DOUBLE_EQ(dist.probability(3.0), 0.5);
    ASSERT_DOUBLE_EQ(dist.probability(4.0), 0.5);
    ASSERT_DOUBLE_EQ(dist.probability(4.0001), 0.0);
}
