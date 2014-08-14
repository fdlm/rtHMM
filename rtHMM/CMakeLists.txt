SET(rtHMM_headers
    decoding.h
    discrete_distribution.h
    distribution.h
    filtering.h
    gaussian_distribution.h
    hmm.h
    mixture_distribution.h
    multidimensional_distribution.h
    uniform_distribution.h
)

SET(rtHMM_implementations
    decoding_impl.h
    filtering_impl.h
    gaussian_distribution_impl.h
    hmm_impl.h
    mixture_distribution_impl.h
    multidimensional_distribution_impl.h
    utils.h
)

ADD_CUSTOM_TARGET(rtHMM SOURCES ${rtHMM_headers} ${rtHMM_implementations})
SET_TARGET_PROPERTIES(rtHMM PROPERTIES PUBLIC_HEADER "${rtHMM_headers}")
