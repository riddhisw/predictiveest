'''
    Package: kf
    Language: Python 2.7

    Modules:
    -------
        common : List of common functions applicable to kf.detailed, kf.fast and
            kf.fast_2 module calculations, with a focus on LKFFB.
        detailed :  LKFFB implementation with detailed data retention of key Kalman
            state variables and derived quantities.
        fast : LKFFB implementation with memoryless Kalman filtering.
        fast_2 : LKFFB implementation leveraging speed of kf.fast (memoryless filtering) but
            retaining some information about state variables.

    Author: Riddhi Gupta <riddhi.sw@gmail.com>
'''
