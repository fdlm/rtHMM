SET(rtHMM_headers
    distribution.h
    hmm.h
    decoding.h
    filtering.h
    observation.h
    observation_cache.h
    discrete_distribution.h
    gaussian_distribution.h
    mixture_distribution.h
    multidimensional_distribution.h
    uniform_distribution.h
)

SET(rtHMM_implementations
    hmm_impl.h
    hmm.cpp
    decoding.cpp
    filtering.cpp
    observation_cache.cpp
    gaussian_distribution_impl.h
    mixture_distribution_impl.h
    multidimensional_distribution_impl.h
    utils.h
)

ADD_LIBRARY(rtHMM STATIC ${rtHMM_implementations})
SET_TARGET_PROPERTIES(rtHMM PROPERTIES PUBLIC_HEADER "${rtHMM_headers}")
