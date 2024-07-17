from abc import ABC, abstractmethod
import pandas as pd
from datetime import timedelta

class Analyser(ABC):
    """
    Abstract class providing an interface for the analyser classifying concept drioft types
    Args:
        config (detector.Config): TBD
        n_samples (int): TBD
    """
    def __init__(self, config, n_samples = 5000):
        self.config = config
        self.n_samples = n_samples
        self.analysis_df = pd.DataFrame()

    @abstractmethod
    def update_window(self, curr_df, curr_date):
        pass

    @abstractmethod
    def decide_class(self, curr_df, curr_date):
        pass

    def calculate_drift(self, ref_df, curr_df):
        feat_drift = []
        for col in self.config.chosen_features:
            feat_drift.append(self.config.drift_test.calculate(ref_df[col], curr_df[col]))
        return self.config.drift_test.calc_drift_strength(feat_drift,self.config.feature_importances)

    def classify_drift(self, curr_df, curr_date):
        self.update_window(curr_df, curr_date)
        return self.decide_class(curr_df, curr_date)

class LastWeekAnalyser(Analyser):
    def update_window(self, curr_df, curr_date):
        cutoff_date = curr_date - timedelta(days=7)
        if (not self.analysis_df.empty) and (self.analysis_df.iloc[0].date <= cutoff_date):
            self.analysis_df = self.analysis_df[self.analysis_df.date >= cutoff_date]

        curr_sample = curr_df.sample(self.n_samples, random_state = 42)
        curr_sample["date"] = curr_date
        self.analysis_df = pd.concat([self.analysis_df,curr_sample])
    
    def decide_class(self, curr_df, curr_date):
        cutoff_date = curr_date - timedelta(days=7)
        if (not self.analysis_df.empty) and (self.analysis_df.iloc[0].date <= cutoff_date):
            drift_yesterday = self.calculate_drift(self.analysis_df[self.analysis_df.date ==
                                                                    curr_date - timedelta(days=1)],curr_df)
            drift_week_ago = self.calculate_drift(self.analysis_df[self.analysis_df.date ==
                                                                    cutoff_date],curr_df)
            drift_last_week = self.calculate_drift(self.analysis_df,curr_df)

            drift_yesterday = self.config.drift_test.sample_drifted(drift_yesterday)
            drift_week_ago = self.config.drift_test.sample_drifted(drift_week_ago)
            drift_last_week = self.config.drift_test.sample_drifted(drift_last_week)

            if(drift_yesterday and drift_week_ago):
                return "Sudden_drift" 
            if((drift_yesterday or drift_last_week) and not drift_week_ago):
                return "Periodic_drift"
            if(drift_last_week):
                return "Incremental_drift"
            if(not drift_yesterday and not drift_last_week):
                return "No_drift"
            
        return "Unknown_drift"