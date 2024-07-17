import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

from detector import test, logger, analyser

class DriftDetector:
    """
    Main class of the detector.

    Args:
        global_config (detector.Config): 
            Configuration used for the global part of the detector
        class_config (detector.Config, optional): 
            Configuration used for the detector's ability to run another set of test on the classes 
            independently to present the information of where the drift happens
        analyser (detector.Analyser, optional): 
            Analyser module for further drift type analysis
        logger (detector.Logger, optional): 
            Logging module to log and plot the detection results through time
    """
    
    def __init__(self, global_config, class_config = None, analyser = None, logger = None):
        self.global_config = global_config
        self.class_config = class_config
        self.analyser = analyser
        self.logger = logger
        self.feature_count = len(global_config.chosen_features)
        self.calculated = False

    
    def detect(self, ref_df, curr_df, curr_date = None, f1 = None):
        """Provide a single round of detection comparing the specified time windows

        Args:
            ref_df (pandas.Dataframe): 
                Reference/historical time window dataset
            curr_df (pandas.Dataframe): 
                Current time window dataset
            curr_date (datetime, optional): 
                Timestamp of the current data sample, used for analyser or logging modules
            f1 (float, optional): 
                F1 score of the model, used for logging or for possible implemetation of error-rate detection in the future
        
        Returns:
            bool: Drift has been detected?
        
        Raises:
            Exception: If current date is not provided in cases where it's mandatory
            Exception: If class drift is wanted, but the class isn't specified in the config
        """
        #Calculate drift for each feature
        feat_drift = []
        for col in self.global_config.chosen_features:
            feat_drift.append(self.global_config.drift_test.calculate(ref_df[col], curr_df[col]))
        self.drift_df = pd.DataFrame(data={"distance": feat_drift}, index=self.global_config.chosen_features)

        #Calculate drift metrics
        self.drift_df["is_drifted"] = self.global_config.drift_test.feature_drifted(self.drift_df.distance)
        self.drift_strength = self.global_config.drift_test.calc_drift_strength(
            self.drift_df.distance, self.global_config.feature_importances)
        self.share_drifted_features = self.drift_df["is_drifted"].sum()/self.feature_count
        self.is_drifted = self.global_config.drift_test.sample_drifted(self.drift_strength)
        self.calculated = True

        #Analysis subroutine
        if(self.analyser is not None):
            self.drift_type = self.analyser.classify_drift(curr_df, curr_date)

        #Class drift subroutine
        if(self.class_config is not None):
            if(self.class_config.class_name is None):
                raise Exception("Class drift is wanted, but the class isn't specified in the config")    
            self.__class_drift(ref_df, curr_df)

        #Logging subroutine
        if(self.logger is not None):
            self.logger.log(curr_date,self.get_drift_statistics(),f1 = f1, class_drift = self.get_class_drift().drift_strength.to_frame().T, feature_drift = self.get_drifted_features(all_features=True).to_frame().T)

        return self.is_drifted
    
    def get_drifted_features(self, all_features = False):
        """Returns the drifted features from the last round of detection
        Args:
            all_features (boolean, optional): 
                Get all feature drifts instead of only the features detected as drifted

        Returns:
            pandas.Series: Sorted series of features concidered as drifted with their corresponding
                drift strengths
        
        Raises:
            Exception: If run before a detectio has been completed
        """

        if(self.calculated):
            if(all_features):
                return(self.drift_df.distance)
            else:
                drifted = self.drift_df[self.drift_df["is_drifted"]].distance
                return drifted.sort_values(ascending = False)
        else:
            raise Exception("No detection has been completed, there are no drift statistics to return.")
        
    def get_drift_statistics(self):
        """Returns the gathered statistics from the last round of detection
             
        Returns:
            dictionary: Dictionary contains whether the last round of detection concidered the sample 
                as drifted, its drift strength and the share of features concidered as drifted. In the
                case of analyser module being present, its drift type decision is returned
        
        Raises:
            Exception: Raised if run before a detection has been completed
        """

        if(self.calculated):
            stats = {
                "is_drifted": self.is_drifted,
                "drift_strength": self.drift_strength,
                "share_drifted_features": self.share_drifted_features
                }
            if(self.analyser):
                stats["drift_type"] = self.drift_type
            return stats
        else:
            raise Exception("No detection has been completed, there are no drift statistics to return.")
        
    def get_class_drift(self):
        """Returns the gathered statistics of the independent tests for each class
             
        Returns:
            dataframe: Dataframe contains the following information for each class test: Whether the last
                round of detection concidered the sample as drifted, its drift strength and the share of 
                features concidered as drifted.
        
        Raises:
            Exception: Raised if run before a detection has been completed or class drift submodule is not present.
        """
        if(self.calculated and self.class_config is not None):
            return self.class_drifts
        else:
            raise Exception("Class drift hasn't been calculated, initialise detector with class_drift = True and run a detect(...)")
        
    def get_test_info(self):
        """Describe the workings of the detector, the tests used and their thresholds, etc.
        Returns:
            string: returned debug information
        """   
        info = "Global test info:\n"
        info += self.global_config.drift_test.info() + "\n\n"
        if(self.class_config is not None):
            info += "Class test info:\n"
            info += self.class_config.drift_test.info() + "\n\n"
        if(self.analyser is not None):
            info += "Drift analyser test info:\n"
            info += self.analyser.config.drift_test.info() + "\n\n"
        if(self.logger is not None):
            info += "Logger is present\n"
        return info


    def __class_drift(self, ref_df, curr_df):
        ref_grouped = ref_df.groupby(self.class_config.class_name)
        ref_class_names = [key for key in ref_grouped.groups.keys()]
        curr_grouped = curr_df.groupby(self.class_config.class_name)
        curr_class_names = [key for key in curr_grouped.groups.keys()]
        class_intersect_names = [i for i in curr_class_names if i in ref_class_names]
        
        drifts = []
        drift_shares = []
        for c in class_intersect_names:
            curr_class = curr_grouped.get_group(c)
            ref_class = ref_grouped.get_group(c)
            feat_drift = []
            for col in self.class_config.chosen_features:
                feat_drift.append(self.class_config.drift_test.calculate(ref_class[col],curr_class[col]))
            feat_drift = np.array(feat_drift)
            drift_shares.append((self.class_config.drift_test.feature_drifted(feat_drift)
                                  .sum())/len(feat_drift))
            drifts.append(self.class_config.drift_test.calc_drift_strength(feat_drift,self.class_config.feature_importances))
        

        self.class_drifts = pd.DataFrame(data={"drift_strength": drifts, "share_drifted": drift_shares}
                                         ,index=[f"Class_{c}" for c in class_intersect_names])
        self.class_drifts["is_drifted"] = self.class_config.drift_test.sample_drifted(self.class_drifts["drift_strength"])
        self.class_drifts = self.class_drifts.sort_values(by="drift_strength")
       


class Config:
    """Configuration of how the individul modules of the detector behave.

    Args:
        chosen_features (list): 
            Subset of features specifying which features to run the test on.
        feature_importances (pandas.Series, optional): 
            Series of feature importances indexed by feature names. Used for calculating the strenght
            of drift, the weighed mean of the individual feature drifts. If not presented, set uniformly
            to result in unweighed meand. Expected to sum up to one.
        drift_test (detector.DriftTest): 
            Drift test to use for the decision of the detector
        class_name (str, optional): 
            Name of the column, where the class labels are present, used for the independend testing of classes
    """

    def __init__(self,chosen_features, feature_importances = None, drift_test = test.WassersteinTest(), class_name = None):
        if(feature_importances is None):
             self.feature_importances = pd.Series([1/len(chosen_features) for i in range(len(chosen_features))], 
                                                 index = chosen_features)           
        else:
            self.feature_importances = feature_importances[chosen_features]

        if(not (len(self.feature_importances.index) == len(chosen_features))):
            raise Exception("Column names specified in the feature importances differ from the chosen features")
        if(not (self.feature_importances.index == chosen_features).all()):
            raise Exception("Column names specified in the feature importances differ from the chosen features")
        self.chosen_features = chosen_features
        self.drift_test = drift_test
        self.class_name = class_name