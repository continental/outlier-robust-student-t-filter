"""
Interface and Implementation of a wide range of various state-of-the-art Bayesian Filtering methods, 
most of them outlier robust filters.
"""

print("-  Loading package src/filters")

from . import abstract, basic, distributions, models, proposed, robust

#from .basic import KalmanFilter
#from .distributions import NormalDistribution, StudentTDistribution, Joint_Indep_Distribution
#from .models import LinearDynamicModel, ExtendedSingerModel
#from .proposed import StudentTFilter

#: A list of the methods implemented in this module
METHODS = {"KalmanFilter":      basic.KalmanFilter,
           "StudentTFilter":    proposed.StudentTFilter,
           "StudentTFilter_GT": proposed.StudentTFilter_GT,
           "Huang_SSM":         robust.Huang_SSM,
           "Agamennoni_VBF":    robust.Agamennoni_VBF,
           "chang_RKF":         robust.chang_RKF,
           "chang_ARKF":        robust.chang_ARKF,
           "Saerkkae_VBF":      robust.Saerkkae_VBF,
           "roth_STF":          robust.roth_STF,
           
           "StudentTFilter_Newton": proposed.StudentTFilter_Newton,
           "StudentTFilter_SF": proposed.StudentTFilter_SF,
           "chang_RKF_SF":      proposed.chang_RKF_SF,

           "KalmanSmoother":    basic.KalmanSmoother,
           "StudentTSmoother":  proposed.StudentTSmoother}