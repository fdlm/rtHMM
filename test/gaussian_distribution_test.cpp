#include <iostream>
#include <sstream>
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "gaussian_distribution.h"

using namespace rtHMM;
using namespace Eigen;
using namespace std;

TEST(gaussian_distribution_test, test_1d_pdf)
{
    gaussian_distribution<1> my_gauss(5.0, 16.0);

    ASSERT_NEAR(my_gauss.probability(5.0), 0.099735570100358176, 0.00001);
    ASSERT_NEAR(my_gauss.probability(10.0), 0.045662271347255479, 0.00001);
    ASSERT_NEAR(my_gauss.probability(0.0), 0.045662271347255479, 0.00001);
    ASSERT_NEAR(my_gauss.probability(8.8), 0.063514764117297243, 0.00001);
}

TEST(gaussian_distribution_test, test_2d_pdf)
{
    Vector2d means2d(3, 5);
    Matrix2d sigma2d;
    sigma2d << 5, 0, 0, 6;

    gaussian_distribution<2> my2dgauss(means2d, sigma2d);
    gaussian_distribution<1> my1dgauss1(3, 5);
    gaussian_distribution<1> my1dgauss2(5, 6);

    Vector2d val(1.324, 3.32);

    ASSERT_NEAR(my2dgauss.probability(val),
                my1dgauss1.probability(val(0)) * my1dgauss2.probability(val(1)),
                0.00001);

}
