#if defined(__linux__) || defined(__MIC__)
#include <time.h>
#define START_TIMER\
    struct timespec rss;\
    clock_gettime(CLOCK_MONOTONIC, &rss);

#define END_TIMER\
    struct timespec rse;\
    clock_gettime(CLOCK_MONOTONIC, &rse);\
    double wtime = (rse.tv_sec - rss.tv_sec)*1000. + (rse.tv_nsec - rss.tv_nsec) / 1000000.;
    //double wtime = (rse.tv_sec - rss.tv_sec)*1000000. + (rse.tv_nsec - rss.tv_nsec) / 1000.;

#elif defined(__APPLE__)
#include <sys/time.h>

#define START_TIMER\
    struct timeval rss;\
    gettimeofday(&rss, 0);

#define END_TIMER\
    struct timeval rse;\
    gettimeofday(&rse, 0);\
    double wtime = (rse.tv_sec - rss.tv_sec)*1000. + (rse.tv_usec - rss.tv_usec)/1000.;
    //double wtime = (rse.tv_sec - rss.tv_sec)*1000000. + (rse.tv_usec - rss.tv_usec);

#endif

#define TIMER wtime
