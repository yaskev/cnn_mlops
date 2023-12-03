# General
This project solves the binary classification problem using Gradient boosting approach with the Titanic dataset.

### Setting up
Run `poetry install`

### Configuration
All configs can be found in `./configs` folder

### Training
Run from root folder: `python mlops/train.py`

### Inference
#### Local
Run from root folder: `python mlops/infer.py`

#### Server
Run `./run_server.sh` and then send requests:

`curl -X POST -H "Content-Type: application/json" -d '{"dataframe_split": {"columns": ["Pclass","Name","Sex","Age","Siblings/Spouses Aboard","Parents/Children Aboard","Fare"], "data": [[1, 1, 1, 29, 1, 9, 50.1]]}}' http://127.0.0.1:5001/invocations`

### Analytics
Run `mlflow ui` and you will be able to see metrics after running training or inference

### Data
Train and test datasets are stored in DVC on Google Drive. When you run train script for the first time, you might be
asked to log in into your account. Should you have any troubles with accessing data, please write me on
Telegram: @yaskev
