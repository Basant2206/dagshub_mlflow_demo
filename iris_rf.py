import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='Basant2206', repo_name='dagshub_mlflow_demo', mlflow=True)



# load dataset
iris=load_iris()
X=iris.data
y=iris.target


# split data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter
max_depth = 5
n_estimators=100
mlflow.set_experiment('iris_dt')
mlflow.set_tracking_uri('https://dagshub.com/Basant2206/dagshub_mlflow_demo.mlflow')
# apply mlflow
with mlflow.start_run():
    rf= RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred= rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param("max_depth", max_depth)

    # create confution metrix
    cm= confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.table('Confusion metrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # save plot
    plt.savefig('confusion_metrix.png')
    mlflow.log_artifact('confusion_metrix.png')
    
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(rf,"random forest")
    mlflow.set_tag("author", "bas")
    mlflow.set_tag("model","Decision tree")
    print("accuracy", accuracy)
