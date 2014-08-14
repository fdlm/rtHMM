SET(rtHMM_test_srcs
    decoding_test.cpp
    discrete_distribution_test.cpp
    gaussian_distribution_test.cpp
    hmm_test.cpp
    filtering_test.cpp
    mixture_distribution_test.cpp
    multidimensional_distribution_test.cpp
    uniform_distribution_test.cpp
)

ADD_EXECUTABLE(runUnitTests ${rtHMM_test_srcs})
TARGET_LINK_LIBRARIES(runUnitTests ${GTEST_BOTH_LIBRARIES})

GTEST_ADD_TESTS(runUnitTests "" ${rtHMM_test_srcs})
