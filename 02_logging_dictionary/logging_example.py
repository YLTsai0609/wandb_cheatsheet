"""

config - Dictionary-like object to save your experiment configuration
         放experiments variable的地方，例如batch size, learning rate,
         preprocessing_method 1, preprocessing_2, dataset_name，但dataset versioning可以使用artifacts
log - Keep track of metrics, video, custom plots, and more - 
    - 控管每個step要給什麼值畫上去，包含x, y，也可以把matplotlib figure傳上去，圖片，聲音，影響，3D檔案，點雲等都可以

This makes us can uplaod anything on the wandb cloud
"""
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd

import wandb

apple = pd.DataFrame(
    {
        "open": np.random.randint(low=0, high=200, size=1000),
        "close": np.random.randint(low=200, high=800, size=1000),
    }
)
# Initialize a new run
wandb.init(project="sample-visualize-predictions", name="metrics")

# Log the metric on each step
for price in apple["close"]:
    wandb.log({"Stock Price": price})

wandb.finish()
