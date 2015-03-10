#ifndef RTHMM_HMM_H
#define RTHMM_HMM_H

#include <vector>
#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "discrete_distribution.h"


namespace rtHMM {

    using namespace std;

    typedef Eigen::SparseMatrix<double> sparse_matrix;
    typedef Eigen::SparseVector<double> sparse_vector;
    typedef Eigen::MatrixXd dense_matrix;
    typedef Eigen::VectorXd dense_vector;

    /*! \brief This class represents an HMM with all its states, transitions and observation distributions.
     *
     *  __IMPORTANT__: Normalisation constraints (such as that probabilities
     *  sum to one) are not checked!
     */
    class hmm {

        public:
            //! \brief Represents a transition to of from another state.
            struct link {
                size_t state_id;    //!< ID of state the transition is to or from
                double probability; //!< Probability of the transition

                bool operator==(const link& rhs) const {
                    return rhs.state_id == state_id;
                }
            };


            /*! \brief Constructs a HMM with a specified number of states
             *
             *  This is the most basic constructor available for an HMM. It
             *  creates an empty HMM that can hold priors, transitions and
             *  observation distributions for the specified number of states.
             *  The prior is __not__ initialised and there are __no__
             *  transitions or observation distributions. The model has no
             *  practical use in this form, and you'll have to set transitions
             *  and observation distributions manually.
             *
             *  \param[in] num_states   Number of states of the HMM
             */
            hmm(size_t num_states)
            {
                prior_probs.resize(num_states);
                successor_links.resize(num_states);
                predecessor_links.resize(num_states);
                observation_dists.resize(num_states);
            }

            /*! \brief Constructs a HMM with a specified prior distribution
             *
             *  This constructor sets a specified prior distribution. The
             *  number of states is determined by the length of the prior. The
             *  model still lacks transitions and observation distributions to
             *  be of any practical use. You'll have to set them manually.
             *
             *  \tparam PT
             *      \parblock
             *      Datatype holding the prior distribution. Supports
             *      rtHMM::sparse_vector, rtHMM::dense_vector and any container
             *      supporting std::begin() and std::end()
             *      \endparblock
             *
             *  \param[in] prior Prior probability distribution of the HMM
             */
            template<typename PT>
            hmm(const PT& prior);

            /*! \brief Constructs a HMM with a specified prior distribution
             *         and transitions.
             *
             *  This constructor sets up the HMM with a prior distribution and
             *  transistions using a transition matrix. You will still have to
             *  set the observation distributions manually.
             *
             *  \tparam PT Datatype holding the prior distribution \sa hmm(const PT& prior)
             *  \tparam TT
             *      \parblock Datatype holding the transitions. Supports
             *      rtHMM::sparse_matrix, rtHMM::dense_matrix and containers of
             *      containers, where both support std::begin() and std::end()
             *      \endparblock
             *
             *  \param[in] prior Prior probability distribution of the HMM
             *  \param[in] transition Transition matrix of the HMM
             *
             *  \pre \paramname{transition} needs to be square
             *  \pre Each dimension of \paramname{transitions} has to be of the
             *       same size as the \paramname{prior}
             */
            template<typename PT, typename TT>
            hmm(const PT& prior, const TT& transition);

            /*! \brief Constructs a HMM with discrete observation
             *         distributions from a prior, transition and observation matrix
             *         and/or sequence container. The i-th row of the observation matrix
             *         is a vector defining the observation probabilities for the i-th
             *         state.
             *
             *  This constructor can process a multitude of types for each of the
             *  parameters. \paramname{prior} and \paramname{transition} can be any
             *  type that the other constructors of the hmm class can handle. The
             *  \paramname{observation} parameter can be either a dense_matrix or
             *  a (sequence) container of (sequence) containers carrying the
             *  probabilities.
             *
             *  \tparam PT type of the parameter holding the prior distribution
             *  \tparam TT type of the parameter holding the transition matrix
             *  \tparam OT type of the parameter holding the observation matrix
             *
             *  \param[in] prior prior probability distribution of the model
             *  \param[in] transition transition matrix of the model
             *  \param[in] observation observation matrix of the model
             */
            template<typename PT, typename TT, typename OT>
            hmm(const PT& prior, const TT& transition, const OT& observation);

            // TODO: add third constructors that accept observation distributions

            /*! \brief Sets the prior probability of a state
             *
             *  \param[in] state_id State ID of the state for which the prior should be set
             *  \param[in] probability Prior probability
             *  \sa prior
             */
            void set_prior(size_t state_id, double probability);

            /*! \brief Sets the transition probability from a state to another state
             *
             *  \param[in] from_state_id State ID of the state from which the transition starts
             *  \param[in] to_state_id State ID of the state to which the transition goes
             *  \param[in] probability Transition probability
             *  \sa successors
             *  \sa predecessors
             */
            void set_transition(size_t from_state_id, size_t to_state_id, double probability);

            /*! \brief Sets the observation distribution for a state
             *
             *  \param[in] state_id State ID of state for which the distribution is set
             *  \param[in] dist Shared pointer to the observation distribution
             *  \sa observation_distribution
             */
            void set_observation_distribution(size_t state_id, shared_ptr<distribution> dist);

            /*! \brief Sets a observation distribution tied to multiple states.
             *
             *  \tparam T  Type of iterable container holding the state ids as size_t.
             *  \param[in] state_ids State IDS sharing the observation distribution
             *  \param[in] dist Shared pointer to the observation distribution
             */
            template<typename T>
            void set_tied_observation_distribution(const T& state_ids, shared_ptr<distribution> dist);

            /*! \brief Gets the prior probability of a state
             *
             *  \param[in] state_id State ID of desired state
             *  \returns Prior probability of state with ID \paramname{state_id}
             *  \sa set_prior
             */
            double prior(size_t state_id) const {
                return prior_probs[state_id];
            }

            /*! \brief Gets links to all successors of a state
             *
             *  \param[in] state_id State ID of desired state
             *  \returns vector with links to all successors of the state
             *  \sa set_transition
             *  \sa link
             */
            const vector<link>& successors(size_t state_id) const {
                return successor_links[state_id];
            }

            /*! \brief Gets links to all predecessors of a state
             *
             *  \param[in] state_id State ID of desired state
             *  \returns vector with links to all predecessors of the state
             *  \sa set_transition
             *  \sa link
             */
            const vector<link>& predecessors(size_t state_id) const {
                return predecessor_links[state_id];
            }

            /*! \brief Gets the observation distribution of a state
             *
             *  \param[in] state_id State ID of desired state
             *  \returns Shared pointer to the observation distribution of the state
             *  \sa set_observation_distribution
             *  \sa set_tied_observation_distribution
             */
            const shared_ptr<distribution> observation_distribution(size_t state_id) const {
                return observation_dists[state_id];
            }

            /*! \brief Gets the number of states of the model
             *
             *  \returns Number of states of the model
             */
            size_t num_states() const {
                return prior_probs.size();
            }

        private:
            vector<double> prior_probs;
            vector<vector<link>> successor_links;
            vector<vector<link>> predecessor_links;
            vector<shared_ptr<distribution>> observation_dists;
    };

} // namespace rtHMM

// include template implementations
#include "hmm_impl.h"

#endif //RTHMM_HMM_H
