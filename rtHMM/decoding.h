#ifndef RTHMM_DECODING_H
#define RTHMM_DECODING_H

#include <array>
#include <list>
#include <vector>

#include "observation_cache.h"

namespace rtHMM {

    using namespace std;

    class hmm;
    class observation;

    /*! \brief This class computes the most probable state sequence from a
     *         defined step in the past until the current observation,
     *         aka fixed-lag decoding using the Viterbi algorithm.
     *
     *  \sa filtering
     */
    class decoding {
        public:
            /*!
             *  \param[in] hmm_model HMM for which decoding should be performed
             *  \param[in] skip_prob
             *      \parblock Threshold below which probability values will be
             *      considered zero. Higher values will result in faster
             *      computations because of sparser filtering distributions,
             *      but with lower accuracy.
             *  \param[in] max_lag
             *      \parblock Maximum number of steps that will be traced
             *      back during backtracking. Minimal value is 1.\endparblock
             *
             *  \sa hmm
             */
            decoding(const hmm& hmm_model, double skip_prob = 0.0, size_t max_lag = 1, bool optimise_tied=false);

            /*! \brief Adds a new observation, resulting in a new decoding step.
             *
             *  \param[in] obs Observation to add.
             */
            void add_observation(const observation& obs);

            /*! \brief Adds a series of observations, resulting in a new decoding steps.
             *
             *  \tparam container_type Type of (iterable) container holding the observations
             *  \param[in] seq Observation to add.
             */
            template<class container_type>
            void add_observation_sequence(const container_type& seq) {
                for (const auto& obs : seq) {
                    add_observation(obs);
                }
            }

            /*! \returns Current length of decoded state sequence */
            size_t n_past_steps() const;

            /*! \returns Maximum length of decoded state sequence that can be stored
             * */
            size_t max_past_steps() const;

            /*! \returns Current viterbi variables */
            const vector<double>& viterbi_variables() const;

            /*! \brief Computes the decoded state sequence
             *
             *  \param[in] back_steps
             *      \parblock Number of states that shall be traced back.
             *      The possible values depend on the \paramname{max_lag}
             *      parameter in the constructor and on how many observations
             *      have already been added.
             *  \sa n_past_steps
             *  \sa max_past_steps
             *
             *  \returns A vector containing the decoded state sequence
             */
            vector<size_t> state_sequence(size_t back_steps) const;

            /*! \brief Computes the decoded state sequence up to the maximum possible length
             *  \returns A vector containing the decoded state sequence
             */
            vector<size_t> state_sequence() const;

            /*! \returns the probability of the decoded state sequence */
            double state_sequence_probability() const;
            /*! \returns the logprobability of the decoded state sequence */
            double state_sequence_log_probability() const;

        private:
            const hmm& model;
            const double skip_prob;
            const size_t memory_size;
            const size_t state_count;

            unique_ptr<internal::observation_cache> obs_prob_calc;
            size_t most_probable_end;

            array<vector<double>, 2> viterbi;
            vector<double>* viterbi_cur;
            vector<double>* viterbi_prev;

            list<vector<size_t>> backtracking_pointers;
            double total_scale_correction;

            vector<size_t> nonzero_elements;
            vector<size_t> new_nonzero_elements;
    };

} // namespace rtHMM

#endif //RTHMM_DECODING_H
