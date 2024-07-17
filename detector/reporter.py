import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from detector import logger, analyser

class Reporter:
    """
    Reporter to generate various statistics and views of the experiment results stored in logs.

    Args:
        TBD
    """
    
    def __init__(self, loggers, experiment_names, chunk_length = None, global_features = ["is_drifted","drift_strength","share_drifted_features","f1"]):
        self.loggers = loggers
        self.experiment_names = experiment_names
        self.chunk_length = chunk_length
        self.log_index = loggers[0].get_logs().index
        self.global_features = global_features


        for logger in loggers:
            if(not (logger.get_logs().index.equals(self.log_index))):
                raise Exception("The indexes of all experiments have to be the same") 
        

    def calculate_main_results(self, logs, index, index_name):
        results = pd.DataFrame()
        for log in logs:
            log = log[self.global_features]
            results = pd.concat([results, log.mean().to_frame().T])
        
        results[index_name] = index
        results = results.set_index(index_name)
        results = results.rename(columns={"is_drifted": "Ratio_of_drift_detections", "drift_strength": "Mean_drift_strength", 
                                                        "share_drifted_features": "Mean_ratio_of_drifted_features", "f1": "Mean_f1_score"})
        return results
    #TODO: Add rozptyl, pocet trid, zkontrolovat ruzne pocty trid
    
    def get_global_results(self):
        return self.calculate_main_results([logger.get_logs() for logger in self.loggers], self.experiment_names, "Experiment name")

    def plot_global_results(self, severity_style, f1_style, detection_style, detection_threshold = None):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax2 = ax.twinx()

        for id, logger in enumerate(self.loggers):
            logs = logger.get_logs()
            ax2.plot(logs.drift_strength,linestyle = severity_style[id]["line"], color = severity_style[id]["color"], 
                     alpha = severity_style[id]["alpha"], label =  f"{self.experiment_names[id]} severity")

            ax.plot(logs.f1, linestyle = f1_style[id]["line"], color = f1_style[id]["color"], 
                    alpha = f1_style[id]["alpha"], label =  f"{self.experiment_names[id]} F1 Score")

            [ ax.axvline(x = detection, linestyle = detection_style[id]["line"], color = detection_style[id]["color"], alpha = detection_style[id]["alpha"],
                         label = f"{self.experiment_names[id]} Drift detection") for detection in logs[logs.is_drifted].index ]

        if detection_threshold: 
            ax2.axhline(y = detection_threshold, color = 'g', linestyle = '--', label = "Drift detection threshold") 

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
        ax.set_xlabel("Date")
        ax.set_ylabel("F1 Score of the model")
        ax2.set_ylabel("Detected drift severity")

        return fig

    def plot_analysis_results(self,detection_threshold = None):
        for id, logger in enumerate(self.loggers):
            print(f"Analysis results of {self.experiment_names[id]} experiment:")
            logs = logger.get_logs()
            print(logs.Drift_type.value_counts())

            fig = plt.figure(figsize = (8,4))
            plt.rc('font', size=12)
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            ax.plot(logs.drift_strength, "b-", alpha = 0.7, label = "Drift severity")
            ax2.plot(logs.f1, "r-", label = "F1 score of the model")
            if(detection_threshold):
                ax.axhline(y = detection_threshold, color = 'g', linestyle = '--', label = "Drift detection threshold") 

            drift_types = list(logs.Drift_type.value_counts().index)
            cmap = plt.cm.get_cmap('Accent', len(drift_types))
            for id, drift_type in enumerate(drift_types):
                for date in logs[logs.Drift_type == drift_type].index :
                    ax2.axvline(x=date, color = cmap(id), linestyle = ':',label = drift_type)

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

            ax.set_ylabel("Drift severity")
            ax.set_xlabel("Date")
            ax.grid(linestyle=':')
            plt.show()

    def get_chunk_results(self, sort_by = "Chunk", ascending = True):
        results = []
        for logger in self.loggers:
            log = logger.get_logs()
            chunks = []
            for i in range(0, len(log), self.chunk_length):
                chunk = log.iloc[i:i+self.chunk_length]
                chunks.append(chunk)
            
            results.append(self.calculate_main_results(chunks, [i for i in range(math.ceil(len(log)/self.chunk_length))], "Chunk"))

        return [result.sort_values(by=sort_by, ascending = ascending) for result in results]
