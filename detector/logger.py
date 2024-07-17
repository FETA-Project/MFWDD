import pandas as pd

class Logger:
    """
    Logger submodule for storing the results of each round of drift detection.
    """
    def __init__(self):
        self.index = []
        self.is_drifted = []
        self.drift_strength = []
        self.share_drifted_features = []
        self.drift_type = []
        self.f1 = []
        self.class_log = pd.DataFrame()
        self.feature_log = pd.DataFrame()

    def log(self,index, drift_statistics, f1 = None, class_drift = None, feature_drift = None):
        """Store the results from the current round of drift detection
        Args:
        date (Integer/Datetime/pd.Timestamp): 
            Time when the test was carried out, either integer index or timestamp is preferable if applicable
        drift_statistics (dictionary): 
            Statistics obtained from detector.get_drift_statistics() 
        f1 (float, optional): 
            F1 score of the model on data from when the test was carried out
        """   
        self.index.append(index)
        self.is_drifted.append(drift_statistics["is_drifted"])
        self.drift_strength.append(drift_statistics["drift_strength"])
        self.share_drifted_features.append(drift_statistics["share_drifted_features"])
        if("drift_type" in drift_statistics):
            self.drift_type.append(drift_statistics["drift_type"])
        if(f1 is not None):
            self.f1.append(f1)
        if(class_drift is not None):
            self.class_log = pd.concat([self.class_log, class_drift])
        if(feature_drift is not None):
            self.feature_log = pd.concat([self.feature_log, feature_drift])

    def get_logs(self, log_type = "global"):
        """Get the stored detection results
        Returns:
            pd.Dataframe: Detection results for each logged date
        """   
        log_data = {
            "is_drifted": self.is_drifted,
            "drift_strength": self.drift_strength,
            "share_drifted_features": self.share_drifted_features
            }
        if(self.drift_type):
            log_data["Drift_type"] = self.drift_type 
        if(self.f1):
            log_data["f1"] = self.f1 
        
        if(log_type == "global"):
            return pd.DataFrame(data=log_data, index = self.index)
        
        if(log_type == "class"):
            return self.class_log.reindex(sorted(self.class_log.columns), axis=1).set_axis(self.index)
        
        if(log_type == "feature"):
            return self.feature_log.reindex(sorted(self.feature_log.columns), axis=1).set_axis(self.index)