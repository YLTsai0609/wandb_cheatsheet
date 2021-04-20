# Wandb

A server/client based experiments management platform. Which makes 

1. your experiments management more efficent.(dashboard)
2. more easiler to sharing your finding.(report)
3. more easiler to compare the result of different trials.(dashboard)

It provides us the ability to track the data pipelines, datasets, and models. 
We can get the report of certain analytics of datasets, data pipeplines. Also the committer.(artifacts)

Wandb seems a tool to make MLops easiler!

[Wandb - Weights and Biases github](https://github.com/wandb/client)

[Wandb - Weights and Biases documentation](https://docs.wandb.ai/)

[Wandb - examples](https://github.com/wandb/examples)
# Installation and login

``` 

pip install wandb
wandb login
```

[My personal page](https://wandb.ai/yltsai0609)

# How it work?

A server is hold by someone.

You are creating data. Sending them to your account on the server.

Then you can visit your data by logging the account.

By

``` Python
wandb.init(
    project="sample-project",
    config={
        "method": 'NORM_1',
        "dropout": 0.2,
        "hidden_layer_size": 128,
        "layer_1_size": 16,
        "layer_2_size": 32,
        "learn_rate": 0.01,
        "decay": 1e-6,
        "momentum": 0.9,
        "epochs": 8}
)

# your_code

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config.epochs,
          callbacks=[WandbCallback(data_type="image", labels=labels)])

# or

wandb.log({"Stock Price": price})

# sending the data to the server.

wandb.finish()
```

There are backup hash-key and log automatically generated when you running wandb trial.

The headline is yulong's question and solution, 

**star** means unsolved.

# Best Practices

1. **Projects**: Log multiple runs to a project to compare them. `wandb.init(project="project-name")`
2. **Groups**: For multiple processes or cross validation folds, log each process as a runs and group them together. `wandb.init(group='experiment-1')`
3. **Tags**: Add tags to track your current baseline or production model.
4. **Notes**: Type notes in the table to track the changes between runs.
5. **Reports**: Take quick notes on progress to share with colleagues and make dashboards and snapshots of your ML projects.

### Advanced Setup

1. [Environment variables](https://docs.wandb.com/library/environment-variables): Set API keys in environment variables so you can run training on a managed cluster.
2. [Offline mode](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): Use `dryrun` mode to train offline and sync results later.
3. [On-prem](https://docs.wandb.com/self-hosted): Install W&B in a private cloud or air-gapped servers in your own infrastructure. We have local installations for everyone from academics to enterprise teams.
4. [Sweeps](https://docs.wandb.com/sweeps): Set up hyperparameter search quickly with our lightweight tool for tuning.
5. [Artifacts](https://docs.wandb.com/artifacts): Track and version models and datasets in a streamlined way that automatically picks up your pipeline steps as you train models.

# 01 keras hello world

There is a wandb callback for keras : 
 `from wandb.keras import WandbCallback`

It's custom for keras model callback argument.

# 02 logging a dictionary

You can log a dictionary-like (python native data structure which can be serielized to JSON) to wandb.

you do actually log json, image and table.

check [Intro notebook](bit.ly/intro-wb)

# *03 exporting data from account to make a local plot

Once we wanna plot a more beautiful plot (like to submit a paper), we need this feature.

# *04 versioning data, model and result across your pipeline

check [here](https://docs.wandb.ai/artifacts)

https://wandb.ai/stacey/deep-drive?workspace=default

# 05 dryrun mode to make sure the wandb setting work

just put environment variable : WANDB_MODE=dryrun

# 06 log a confusition matrix on wandb dashboard

extend the callback and use it.

[check this report](https://wandb.ai/mathisfederico/wandb_features/reports/Better-visualizations-for-classification-problems--VmlldzoxMjYyMzQ)

check the doc of 03_custom_callback

we can modify the callback so that we can log any metrics we want including(dataframe)

[example](https://github.com/YLTsai0609/bert_ner/blob/main/f1_wandbcallback.py) : log precision, recall, f1-score by integraing EvalCallback in `kashgari` and WandbCallback


# 使用心得

## Good
[ ] - (推) 可輕易比較Cross-run的performance，code version，data version，terminal message，hyperparameters
[ ] - 可realtime追蹤訓練進度(streamming by per repch / smaller checkpoint)
[ ] - 可基於已知的hyperparameter畫圖，協助推斷最佳hyperparameter

## Bad

1. callback - Image的支援比較多，包含(隨著模型訓練，觀察樣本的預測)
2. callback - 要做數值外的分析，比較複雜，要繼承`WandbCallback`自己實作想要的內容，想辦法用plotly呈現，會花費額外開發時間
3. model-saving-checkpoint - 很多時候測試SOTA模型(非標準模型，都有客製化object)，目前還沒看到從哪裡改callback比較快