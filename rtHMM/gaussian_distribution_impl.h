namespace rtHMM {

    static const double pi = 4.0 * std::atan(1.0);

    template<int dims>
    gaussian_distribution<dims>::gaussian_distribution(const mean_type& mn, const variance_type& var) :
        means(mn),
        inverse_sigma(var.inverse())
    {
        compute_norm_factor(var);
    }

    template<int dims>
    double gaussian_distribution<dims>::compute_probability(const value_type& value) const
    {
        auto dev = value - means;
        double tmp = (dev.transpose() * inverse_sigma) * dev;
        return norm_factor * std::exp(-0.5f * tmp);
    }

    template<int dims>
    void gaussian_distribution<dims>::compute_norm_factor(const variance_type& var)
    {
        norm_factor = 1 / (std::pow(2. * pi, static_cast<double>(dims) / 2) * std::sqrt(var.determinant()));
    }

    inline gaussian_distribution<1>::gaussian_distribution(double mn, double var) :
        mean(mn),
        two_times_variance_inverse(1. / (var * 2.)),
        norm_factor(1. / sqrt(2. * var* pi))
    {
    }

    inline gaussian_distribution<1>::gaussian_distribution(const gaussian_distribution<1>::mean_type& mn,
            const gaussian_distribution<1>::variance_type& var) :
        gaussian_distribution(mn(0, 0), var(0, 0))
    {
    }

    inline double gaussian_distribution<1>::compute_probability(const double& value) const
    {
        return norm_factor * exp(-pow(value - mean, 2) * two_times_variance_inverse);
    }

} // namespace rtHMM
