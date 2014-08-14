#ifndef RTHMM_GAUSSIAN_DISTRIBUTION_H
#define RTHMM_GAUSSIAN_DISTRIBUTION_H

#include <cmath>
#include <Eigen/Dense>
#include "distribution.h"

namespace rtHMM {

    /*! \brief A possibly multi-dimensional Gaussian distribution.
     *  \tparam dims Number of dimensions
     *  \sa distribution
     */
    template<int dims>
    class gaussian_distribution : public distribution<Eigen::Matrix<double, dims, 1>> {
        public:
            /*! \brief Covariance matrix type */
            typedef Eigen::Matrix<double, dims, dims> variance_type;
            /*! \brief Mean vector type */
            typedef Eigen::Matrix<double, dims, 1> mean_type;
            /*! \brief Convenience typedef, value has the same type as the mean */
            typedef mean_type value_type;

            /*! \brief Constructs the Gaussian with a given mean vector and
             *         covariance matrix. Without parameters, all dimensions are
             *         treated as independent with means of 0 and variances of
             *         1.
             *  \param[in] mean Mean values for each dimension
             *  \param[in] variance Covariance matrix
             */
            gaussian_distribution(const mean_type& mean = mean_type::Zero(), const variance_type& variance = variance_type::Identity());

        private:
            mean_type means;
            variance_type inverse_sigma;
            double norm_factor;

            void compute_norm_factor(const variance_type& var);
            double compute_probability(const value_type& value) const override;
    };

    /*! \brief A one-dimensional Gaussian distribution.
     *  \sa distribution
     */
    template<>
    class gaussian_distribution<1> : public distribution<double> {
        public:
            /*! \brief Variance is just a single value */
            typedef Eigen::Matrix<double, 1, 1> variance_type;
            /*! \brief Mean is just a single value */
            typedef Eigen::Matrix<double, 1, 1> mean_type;

        public:
            /*! \brief Constructs the Gaussian with a given mean and variance.
             *  \param[in] mean Mean of the distribution
             *  \param[in] variance Variance of the distribution
             */
            gaussian_distribution(double mean = 0.f, double variance = 1.f);
            /*! \brief Constructs the Gaussian with a given mean and variance.
             *         Here, mean and variance can be passed as a single-valued
             *         matrix/vector for compatibility reasons.
             *  \param[in] mean Mean of the distribution
             *  \param[in] variance Variance of the distribution
             */
            gaussian_distribution(const mean_type& mean, const variance_type& variance);

        private:
            double mean;
            double two_times_variance_inverse;
            double norm_factor;

            double compute_probability(const double& value) const override;
    };

} // namespace rtHMM

// include template implementations
#include "gaussian_distribution_impl.h"

#endif // RTHMM_GAUSSIAN_DISTRIBUTION_H
