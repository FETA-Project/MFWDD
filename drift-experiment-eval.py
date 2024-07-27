import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cesnet_datazoo.datasets import CESNET_TLS_Year22
from cesnet_datazoo.config import DatasetConfig, AppSelection
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import csv
import json
import numpy as np
import pandas as pd
from detector.detector import DriftDetector, Config
from detector.logger import Logger
from detector.test import KSTest, WassersteinTest
from detector.analyser import LastWeekAnalyser

import math
from scipy import stats

from sklearn.metrics import f1_score

from joblib import load

import warnings
warnings.filterwarnings('ignore') 


##### Support Functions
def get_long_flow(ppi):
    tmp_cnt = 0
    for i in range(1,31):
        if ppi["DIR_"+str(i)] == 0:
            tmp_cnt += 1
    ppi_len = 30 - tmp_cnt
    return ppi_len

def get_long_flow_nettisa(ppi):
    PPI_DIR = 0
    ppi_len = 30 - np.count_nonzero(ppi[PPI_DIR] == 0)
    return ppi_len


class NettisaConvertor:
    def __init__(self):
        pass

    def get_nettisa(self, ppi: np.ndarray):
        PPI_IPT = 0
        PPI_DIR = 1
        PPI_SIZE = 2
        PPI_PUSH_FLAG = 3

        ppi_len = 30 - np.count_nonzero(ppi[PPI_DIR] == 0)
        nts_mean = np.mean(ppi[PPI_SIZE][:ppi_len])
        nts_min = ppi[PPI_SIZE][:ppi_len].min()
        nts_max = ppi[PPI_SIZE][:ppi_len].max()
        nts_std = np.std(ppi[PPI_SIZE][:ppi_len])
        nts_rts = np.sqrt(np.mean(ppi[PPI_SIZE][:ppi_len]**2))
        nts_avg_dis = np.mean(np.abs(ppi[PPI_SIZE][:ppi_len] - nts_mean))
        nts_kurtosis = np.sum((ppi[PPI_SIZE][:ppi_len] - nts_mean)**4) / (ppi_len * nts_std**4)
        relative_times = np.cumsum(ppi[PPI_IPT][:ppi_len])
        nts_mean_relative_time = np.mean(relative_times)
        nts_mean_time_differences = np.mean(ppi[PPI_IPT][:ppi_len])
        nts_min_time_differences = ppi[PPI_IPT][:ppi_len].min()
        nts_max_time_differences = ppi[PPI_IPT][:ppi_len].max()
        nts_time_distribution = ((np.sum(np.abs(nts_mean_time_differences - ppi[PPI_IPT][:ppi_len]))) / (ppi_len-1)) / ((nts_max_time_differences - nts_min_time_differences) / 2)
        n_swiches = np.count_nonzero(np.diff(ppi[PPI_DIR][:ppi_len]))
        nts_switching_ratio = n_swiches / ( (ppi_len - 1) / 2)
        nts_max_minus_min = nts_max - nts_min
        nts_percent_deviation = nts_avg_dis / nts_mean
        nts_variance = nts_std**2
        nts_burtines = (nts_std - nts_mean) / (nts_std + nts_mean)
        nts_coef_variation = nts_std / nts_mean
        return  nts_mean, nts_min, nts_max, nts_std, nts_rts, nts_avg_dis, nts_kurtosis, nts_mean_relative_time, nts_mean_time_differences, nts_min_time_differences, nts_max_time_differences, nts_time_distribution, nts_switching_ratio, nts_max_minus_min, nts_percent_deviation, nts_variance, nts_burtines, nts_coef_variation

    def update_df_with_nettisa_features(self,df):
        results = df['PPI'].apply(lambda x: self.get_nettisa(x))
        df_results = pd.DataFrame(results.tolist(), columns=['nts_mean', 'nts_min', 'nts_max', 'nts_std', 'nts_rts', 'nts_avg_dis', 'nts_kurtosis', 'nts_mean_relative_time', 'nts_mean_time_differences', 'nts_min_time_differences', 'nts_max_time_differences', 'nts_time_distribution', 'nts_switching_ratio', 'nts_max_minus_min', 'nts_percent_deviation', 'nts_variance', 'nts_burtines', 'nts_coef_variation'])
        df = pd.concat([df, df_results], axis=1)
        df['nts_directions'] = df.PACKETS / (df.PACKETS + df.PACKETS_REV)
        df = df.drop(columns=["PPI"])
        #print("returning",df)
        return df

def get_data_window(nettisa, filter_short_flows, data):
    # Convert to Nettisa
    if not filter_short_flows:
        if nettisa:
            data_ref = data.get_test_df()
            data_ref = ntc.update_df_with_nettisa_features(data_ref)
        else:
            data_ref = data.get_test_df(flatten_ppi=True)
    # Filter Short Flows
    else:
        if nettisa:
            data_ref = data.get_test_df()
            data_ref = data_ref[ data_ref["PPI"].apply(get_long_flow_nettisa) < 30 ]
            data_ref = ntc.update_df_with_nettisa_features(data_ref)
            #print(len(data_ref))
        else:
            data_ref = data.get_test_df(flatten_ppi=True)
            data_ref = data_ref[ data_ref.apply(get_long_flow, axis=1) < 30 ]
        for key,item in data_ref["APP"].value_counts().items():
            if item < 2:
                filtered_df = data_ref[data_ref['APP'] == key]
                data_ref.drop(filtered_df.index , inplace=True)
    return data_ref

##### Configure Experiment
VERBOSE=0
FILTER_SHORT_FLOWS=0
NETTISA=0
ntc = NettisaConvertor()
LOG_FILE = "drift-metadata-tls-ppi-short.txt"

##### Run Functions
Xdata = None
ydata = None
feat_names = None
label_name = "APP" 
ref_clf = None 

## BEGIN Prep DataZoo
data = CESNET_TLS_Year22("/home/dosoukup/Dataset/Thesis_contents/testing/datasets/TLS/", size="XS")
common_params = {
    "dataset": data,
    "apps_selection": AppSelection.ALL_KNOWN,
    "train_period_name": "W-2022-1",
    "use_packet_histograms": True,
    "use_tcp_features": True
}

hist_df = pd.DataFrame()
current_date = datetime(2022, 1, 1)
while current_date <= datetime(2022, 1, 7):
    dataset_config = DatasetConfig(**common_params, test_period_name=current_date.strftime("M-%Y-%m"), test_dates=[current_date.strftime("%Y%m%d")])
    data.set_dataset_config_and_initialize(dataset_config)
    #curr_df = data.get_test_df(flatten_ppi=True)
    curr_df = get_data_window(NETTISA, FILTER_SHORT_FLOWS, data)

    curr_sample = curr_df.sample(10000, random_state = 42, replace=True)
    curr_sample["date"] = current_date
    hist_df = pd.concat([hist_df,curr_sample])
    current_date += timedelta(days=1)

feat_names = dataset_config.get_feature_names(flatten_ppi=True)
if NETTISA:
    feat_names = hist_df.keys().drop("APP","date")
    feat_names = feat_names.drop("date")

Xdata = hist_df.drop(columns=["APP","date"])
# TODO change to universal selection
ydata = hist_df.APP
X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.33, random_state=42,stratify=ydata)

from sklearn.preprocessing import LabelEncoder
# TODO improve -> datazoo brat vice dat nebo ty problematicke tridy vyhodit -> respektive sesbirat list problematickych trid
#encoder=LabelEncoder()
#y_train=encoder.fit_transform(y_train)
ref_clf = XGBClassifier().fit(X_train, y_train)
y_pred = ref_clf.predict(X_test)

print(f"F1 Base Score: {f1_score(y_test, y_pred, average = 'weighted')}")
## END Prep DataZoo

## BEGIN Test definition
#Configure the global test providing detection on the whole sample
global_config = Config(
    chosen_features = feat_names,
    feature_importances = pd.Series(ref_clf.feature_importances_,index = ref_clf.feature_names_in_),
    drift_test=KSTest(drift_threshold_global=0.475, drift_threshold_single = 0.05)
)

# Configure the test used for the independend class detections
class_config = Config(
    chosen_features = feat_names,
    feature_importances = pd.Series(ref_clf.feature_importances_,index = ref_clf.feature_names_in_),
    drift_test=WassersteinTest(),
    class_name=label_name
)

#Configure the analyser for classification of different drift types
analyser_config = Config(
    chosen_features = feat_names,
    feature_importances = pd.Series(ref_clf.feature_importances_,index = ref_clf.feature_names_in_),
    drift_test=WassersteinTest(drift_threshold_global=0.05)
)

#Logger of the single reference model without retraining
logger_ref = Logger()
detector_ref = DriftDetector(global_config,class_config, logger=logger_ref)
analyser_ref = LastWeekAnalyser(analyser_config)

#Loggers for the retrained models
loggers = [Logger()]
detector_drift = DriftDetector(global_config,class_config, logger=loggers[0])
analyser = LastWeekAnalyser(analyser_config)
## END Test definition


f1_no_retraining = []
f1_drift_retraining_sum = []
f1_drift_retraining = []

retraining_clf = ref_clf

ref_df = hist_df.copy(deep=True)

# TODO prepare several test scenarios
# TODO switch time vs samples
# TODO switch for class/global drift
current_date = datetime(2022, 1, 8)
while current_date <= datetime(2022, 5, 30):
    try:
        metadata = {"date": current_date}
        #Get current data
        dataset_config = DatasetConfig(**common_params, test_period_name=current_date.strftime("M-%Y-%m"), test_dates=[current_date.strftime("%Y%m%d")])
        data.set_dataset_config_and_initialize(dataset_config)
        test_df = get_data_window(NETTISA, FILTER_SHORT_FLOWS, data)
        
        #Test model with no retraining
        Xdata = test_df.drop(columns=[label_name])
        # TODO change to universal selection
        ydata = test_df.APP
        
        y_pred = ref_clf.predict(Xdata)
        f1_no_retraining.append(f1_score(ydata,y_pred, average = 'weighted'))
        # TODO check input dataframe variants
        #detector_ref.detect(ref_df,Xdata,current_date,f1_score(ydata,y_pred, average = 'weighted'))
        detector_ref.detect(ref_df,test_df,current_date,f1_score(ydata,y_pred, average = 'weighted'))
        metadata["f1_no_retrain"] = f1_score(ydata,y_pred, average = 'weighted')
        drift_result = detector_ref.get_drift_statistics()
        metadata["global"] = {"non-retrain": None}
        metadata["global"]["non-retrain"] = {"drifted_features_ration": drift_result["share_drifted_features"], "drifted_features": len(detector_ref.get_drifted_features()), "drift_strenght": drift_result["drift_strength"]}
        
        ## per class data drift info
        #class_drift_result = detector_ref.get_drift_statistics()
        #metadata["class"] = {"non-retrain":None}
        #metadata["class"]["non-retrain"] = {"drifted_classes":len(class_drift_result[ class_drift_result["is_drifted"]==True ]), "all_classes": len(class_drift_result["is_drifted"])}


        #Test retraining model
        y_pred = retraining_clf.predict(Xdata)
        f1_drift_retraining.append(f1_score(ydata,y_pred, average = 'weighted'))
        ## TODO check input dataframes
        #is_drifted = detector_drift.detect(hist_df,Xdata,current_date,f1_score(ydata,y_pred, average = 'weighted'))
        is_drifted = detector_drift.detect(hist_df,test_df,current_date,f1_score(ydata,y_pred, average = 'weighted'))

        metadata["f1_retrain"] = f1_score(ydata,y_pred, average = 'weighted')
        metadata["is_drifted"] = is_drifted
        drift_result = detector_drift.get_drift_statistics()
        #metadata["global"] = {"retrain": None}
        metadata["global"]["retrain"] = {"drifted_features_ration": drift_result["share_drifted_features"], "drifted_features": len(detector_drift.get_drifted_features()), "drift_strenght": drift_result["drift_strength"]}
        print("TMP global drift",drift_result)

        ## per class data drift
        #class_drift_result = detector_drift.get_class_drift()
        #print("TMP class drift",class_drift_result)
        #metadata["class"] = {"drifted":None}
        #metadata["class"]["drifted"] = {"drifted_classes":len(class_drift_result[ class_drift_result["is_drifted"]==True ]), "all_classes": len(class_drift_result["is_drifted"])}
        metadata["classes"] = len(ydata.value_counts())

        if is_drifted:
        #    print("1",loggers)
            loggers.append(Logger())
            detector_drift = DriftDetector(global_config, logger=loggers[-1])
            detector_drift.detect(hist_df,Xdata,current_date,f1_score(ydata,y_pred, average = 'weighted'))
            print("Drift detected, retraining")

            #Update training dataset
            #print("taking", len(hist_df)-len(test_df))
            if len(hist_df) < len(test_df):
                hist_df = pd.DataFrame()
            else:
                hist_df = hist_df.tail(len(hist_df)-len(test_df))
            test_df["date"] = current_date
            hist_df = pd.concat([hist_df,test_df])
            Xdata = hist_df.drop(columns=[label_name,"date"])
            # TODO universal selection
            ydata = hist_df.APP
            retraining_clf = XGBClassifier().fit(Xdata, ydata)
        # Write metadata to file
        sample = open(LOG_FILE, 'a')
        print(metadata, file = sample)
        sample.close()

    except Exception as error:
        print("An error occurred:", error, "on",current_date)
    current_date += timedelta(days=1)

for l in loggers:
    sample = open('tls-drft-detection.txt', 'a')
    print(l.get_logs(), file = sample)
    sample.close()


dates = [l.get_logs().index.values for l in loggers]

plt.figure(figsize=(12,6))
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(12, 5))
ax2 = ax.twinx()
logs = logger_ref.get_logs()
plt.rcParams.update({'font.size': 12})

ax2.plot(logs.drift_strength,"r--",alpha = 0.5, label =  "Reference drift severity")
ax.plot(logs.f1, "b--", alpha = 0.5, label = "Reference F1 score")

for i, logger in enumerate(loggers):
    logs = logger.get_logs()
    ax2.plot(logs.drift_strength,"r-", alpha = 0.6, label =  "Retrained model drift severity")
    ax.plot(logs.f1,"b-",alpha = 0.6, label =  "Retrained model F1 score")


for detection in [dates[i][-1] for i in range(len(dates)-1)]:
    ax.axvline(x = detection, color = 'y', alpha = 0.5, label = 'Drift detection')
ax2.axhline(y = 0.05, color = 'g', linestyle = '--', label = "Drift detection threshold") 

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
handles, labels = ax2.get_legend_handles_labels()
by_label = by_label | dict(zip(labels, handles))

pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.9])
ax.grid(linestyle=':')
ax.legend(
    by_label.values(),
    by_label.keys(),
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15),
    fancybox=True,
    ncol=3, 
)
ax.set_ylim(0.4,1)
ax.set_xlabel("Date")
ax.set_ylabel("F1 Score of the model")
ax2.set_ylabel("Detected drift severity")
plt.savefig('drift-tls-since-may-long-lower-threshold.png',bbox_inches='tight')
