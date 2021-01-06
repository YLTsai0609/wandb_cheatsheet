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

# 01 keras hello world

There is a wandb callback for keras : 
 `from wandb.keras import WandbCallback`

It's custom for keras model callback argument.

# 02 logging a dictionary

You can log a dictionary-like (python native data structure which can be serielized to JSON) to wandb.

# 03 exporting data from account to make a local plot

Once we wanna plot a more beautiful plot (like to submit a paper), we need this feature.
