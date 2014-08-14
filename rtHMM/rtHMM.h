#ifndef RTHMM_RTHMM_H
#define RTHMM_RTHMM_H

#define RTHMM_VERSION "0.1.0"

#include "hmm.h"

#include "decoding.h"
#include "filtering.h"

#include "distribution.h"
#include "discrete_distribution.h"
#include "gaussian_distribution.h"
#include "uniform_distribution.h"
#include "mixture_distribution.h"
#include "multidimensional_distribution.h"

/*! \brief This is the namespace of the rtHMM library. */
namespace rtHMM {
    /*! \brief This is an internal namespace for stuff the libary uses but
     *         that is not directly usable to the user.
     */
    namespace internal {
    }
}

#endif //RTHMM_RTHMM_H
