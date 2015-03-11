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
    rtHMM.h
    uniform_distribution.h
    utils.h
)

SET(rtHMM_header_implementations
    hmm_impl.h
    gaussian_distribution_impl.h
    mixture_distribution_impl.h
    multidimensional_distribution_impl.h
    )

SET(rtHMM_implementations
    hmm.cpp
    decoding.cpp
    filtering.cpp
    observation_cache.cpp
)

ADD_LIBRARY(rtHMM STATIC ${rtHMM_implementations})

#TODO: make this work for windows, too!
SET(LIB_TYPE LIBRARY)
SET(RTHMM_INCLUDE_PATH /usr/local/include/rtHMM)
SET(CMAKE_INSTALL_PREFIX /usr/local/lib/)

INSTALL(TARGETS rtHMM 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}
)
INSTALL(FILES ${rtHMM_headers} ${rtHMM_header_implementations} DESTINATION ${RTHMM_INCLUDE_PATH})
