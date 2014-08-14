#ifndef RTHMM_FILTERING_H
#define RTHMM_FILTERING_H

#include <list>
#include <vector>
#include <limits>

#include "hmm.h"
#include "utils.h"
#include "observation_cache.h"

namespace rtHMM {

    using namespace std;

    /*! \brief This class computes the filtering distribution using the
     *         forward algorithm.
     *
     *  \sa decoding
     *
     *  \tparam hmm_type Type of the HMM for which the filtering distribution will be computed
     *  \tparam optimise_tied
     *      \parblock If true, observation probabilities of tied states are
     *      computed only once per step. This introduces a small overhead, but can
     *      speed up the overall computation a lot. If false, the probability of tied states
     *      is computed repeatedly for each state. If you have many tied states,
     *      set to true. \endparblock
     */
    template<typename hmm_type, bool optimise_tied = false>
    class filtering {
        public:
            /*!
             *  \param[in] hmm_model HMM for which the filtering distribution will be computed
             *  \param[in] skip_prob
             *      \parblock Threshold below which probability values will be
             *      considered zero. Higher values will result in faster
             *      computations because of sparser filtering distributions,
             *      but with lower accuracy.
             *  \param[in] max_past_steps
             *      \parblock Maximum number of previous distributions
             *      that will be saved. The minimal value is 1. \endparblock
             *
             *  \sa hmm
             */
            filtering(const hmm_type& hmm_model, double skip_prob = 0.0, size_t max_past_steps = 1);

            /*! \brief Adds a new observation, resulting in a new filtering step.
             *
             *  \param[in] obs Observation to add.
             */
            void add_observation(const typename hmm_type::observation_type& obs);

            /*! \brief Adds a series of observations, resulting in a new filtering steps.
             *
             *  \tparam container_type Type of (iterable) container holding the observations
             *  \param[in] seq Observation to add.
             */
            template<class container_type>
            void add_observation_sequence(const container_type& seq);

            /*! \returns Number of past distributions currently stored */
            size_t n_past_steps() const;

            /*! \returns Maximum number of past distribution that can be stored
             * */
            size_t max_past_steps() const;

            /*! \brief Compute the sequence probability
             *  \returns Probability of the sequence observed so far.
             */
            double sequence_probability() const;

            /*! \brief Compute the logarithm of the sequence probability
             *  \returns Logarithm of the probability of the sequence observed so far.
             */
            double sequence_log_probability() const;

            /*! \brief Yields the filtering distribution
             *
             *  \param[in] back_steps
             *      \parblock Number of steps in the past for which the filtering
             *      distribution should be returned. The possible values depend
             *      on the \paramname{n_past_steps} parameter in the constructor
             *      and on how many observations have already been added.
             *  \sa filtering
             *  \sa n_past_steps
             *  \sa max_past_steps
             *
             *  \returns A vector containing the probabilities for being in each state
             */
            const vector<double>& distribution(size_t back_steps = 0) const;

        private:
            const hmm_type& model;
            const size_t state_count;
            const double skip_prob;
            const size_t memory_size;

            double seq_prob;
            typename internal::cache_type<optimise_tied, hmm_type>::type obs_calc;

            list<vector<double>> alpha;

            vector<size_t> nonzero_elements;
            vector<size_t> new_nonzero_elements;

    };

} //namespace rtHMM


// Implementation of template methods
#include "filtering_impl.h"

#endif //RTHMM_FILTERING_H
