'''
    Package: analysis_tools
    Language: Python 2.7

    Modules:
    -------
        common : List of functions required by multiple modules in analysis_tools.
        experiment : Maintain experimentally controlled parameter regimes and
            derived quantities.
        truth : Defines true stochastic state and its properties.
        noisydata : Simulates noisy experimental data based on a engineered truth
            dclass instance, defining a dephasing noise field.
        kalman : Execute LKFFB runs for a given Experiment and Noisy Data instance.
        riskanalysis : Optimises LKFFB Kalman noise variance parameters for a parameter regime
            specified by (testcase, variation); and used Bayes Risk metric to assess
            predictive performance.

    Author: Riddhi Gupta <riddhi.sw@gmail.com>
'''
