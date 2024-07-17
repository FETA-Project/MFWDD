from scipy import stats
from scipy.spatial import distance
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class DriftTest(ABC):
    """
    Abstract class providing an interface for the statistical drift tests
    Args:
        drift_threshold_single (float): Threshold for single feature drift detection, after which is the single feature
            concidered drifted
        drift_threshold_global (float): Threshold for weighed global drift strength, after which detector concideres 
            the current sample as drifted and returns a detection
    """
    def __init__(self, drift_threshold_single, drift_threshold_global):
        self.drift_threshold_single = drift_threshold_single
        self.drift_threshold_global = drift_threshold_global

    @abstractmethod
    def calculate(self, ref, curr):
        """Execute test on historical and current windows
        Args:
        ref (1D array/pd.Series): 
            Reference/Historical subset of data corresponding to one feature
        curr (1D array/pd.Series): 
            Current subset of data corresponding to one feature
        Returns:
            float: Value of the test result
        """        
        pass
    
    @abstractmethod
    def feature_drifted(self, metrics):
        """Annotate a series of test scores with whether the features should be concidered drifted
        Args:
        metrics (pd.Series): 
            Test result metrics to anotate whether the features should be concidered drifted
        Returns:
            pd.Series: Bool for each feature whether it is drifted or not
        """       
        pass

    @abstractmethod
    def sample_drifted(self, metric):
        """Decide by the calculated drift strength whether the whole sample is drifted
        Args:
        metric (float): 
            Calculated drift strength
        Returns:
            bool: Specifies whether the whole smaple is concidered drifted
        """   
        pass

    @abstractmethod
    def calc_drift_strength(self, drifts, weights):
        """Calculate the final drift strength/severity based on the individual test results
        Args:
        drifts (float): 
            Calculated drift strength
        Returns:
            bool: Specifies whether the whole smaple is concidered drifted
        """   
        pass
    
    @abstractmethod
    def info(self):
        """Describe the workings and thresholds of the test
        Returns:
            string: returned debug information
        """   
        pass
 
class WassersteinTest(DriftTest):
    def __init__(self, drift_threshold_single = 0.1, drift_threshold_global = 0.05):
        super().__init__(drift_threshold_single, drift_threshold_global)

    def calculate(self, ref, curr):
        if stats.tstd(ref) == 0:
            if ref.iloc[0] == curr.iloc[0]:
                return 0
            return 1
        return stats.wasserstein_distance(ref,curr)/stats.tstd(pd.concat([ref,curr]))
    
    def feature_drifted(self, metrics):
        return metrics > self.drift_threshold_single

    def sample_drifted(self, metric):
        return metric > self.drift_threshold_global
    
    def calc_drift_strength(self, drifts, weights):
        #Weight normalisation
        weights = weights/weights.sum()
        return (drifts*weights).sum()
    
    def info(self):
        return f"""Wasserstein test compares the normalised Wasserstein distance between historical and current distribution.
A feature is concidered drifted when the distance is higher than {self.drift_threshold_single}..
The whole sample is concidered drifted when the weighed mean of all test results is higher than {self.drift_threshold_global}."""     

class KSTest(DriftTest):
    def __init__(self, drift_threshold_single = 0.05, drift_threshold_global = 0.475):
        super().__init__(drift_threshold_single, drift_threshold_global)

    def calculate(self, ref, curr):
        return stats.ks_2samp(ref, curr).pvalue
    
    def feature_drifted(self, metrics):
        return metrics < self.drift_threshold_single

    def sample_drifted(self, metric):
        return metric < self.drift_threshold_global
    
    def calc_drift_strength(self, drifts, weights):
        weights = np.array(weights)
        weights = (1-weights)/(1-weights).sum() #Invert and normalise weights
        return (drifts*weights).sum()
    
    def info(self):
        return f"""KS test uses the two-sample kolmogorov-smirnov test to decide whether the historical and current distributions differ. 
A feature is concidered drifted when p-value is lower than {self.drift_threshold_single}.
The whole sample is concidered drifted when the weighed mean of all test results is lower than {self.drift_threshold_global}."""        

class JSTest(DriftTest):
    def __init__(self, drift_threshold_single = 0.1, drift_threshold_global = 0.05, n_bins = 25):
        super().__init__(drift_threshold_single, drift_threshold_global)
        self.n_bins = n_bins

    def calculate(self, ref, curr):
        bin_min = min(min(ref),min(curr))
        bin_max = max(max(ref),max(curr))
        h_ref = np.histogram(ref,bins=self.n_bins, range=(bin_min,bin_max))
        h_curr = np.histogram(curr,bins=self.n_bins, range=(bin_min,bin_max))
        return distance.jensenshannon(h_ref[0],h_curr[0])
    
    def feature_drifted(self, metrics):
        return metrics > self.drift_threshold_single

    def sample_drifted(self, metric):
        return metric > self.drift_threshold_global
    
    def calc_drift_strength(self, drifts, weights):
        return (drifts*weights).sum()
    
    def info(self):
        return f"""JS test compares the Jensen-Shannon distance between historical and current distribution.
A feature is concidered drifted when the distance is higher than {self.drift_threshold_single}..
The whole sample is concidered drifted when the weighed mean of all test results is higher than {self.drift_threshold_global}.
To transform the distribution to probability vectors, binning with {self.n_bins} bins is done."""   