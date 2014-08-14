#ifndef HMM_MATRICES_H
#define HMM_MATRICES_H

#define STATE_COUNT 3
#define ALPHABET_SIZE 3
#define ALPHABET { 0ul, 1ul, 2ul }
#define PRIOR 0.6, 0.2, 0.2
#define TRANSITION 0.7, 0.3, 0.0, \
                   0.1, 0.6, 0.3, \
                   0.0, 0.3, 0.7

#define OBSERVATION 0.7, 0.15, 0.15, \
                    0.3, 0.5,  0.2, \
                    0.2, 0.4,  0.4


#endif //HMM_MATRICES_H
