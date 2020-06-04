import click
import joblib
import pandas as pd
import numpy as np


from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from utils import save_model


@click.group()
def run():
    pass


def define_parameters_gb():
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 7, 10]
    min_samples_leaf = [1, 2, 3]
    loss = ['deviance']
    learning_rate = [0.1, 0.05, 0.01]
    param_grid = {'loss' : loss,
                  'n_estimators' : n_estimators,
                  'learning_rate': learning_rate,
                  'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  'max_features': max_features,
                  'min_samples_split': min_samples_split
                 }
    return param_grid


def do_hyperparametrization(classifier, param_grid, X_train, y_train, n_iter, 
                            kfold, metric, n_jobs, verbose):
    rs_clf = RandomizedSearchCV(classifier, param_distributions=param_grid, 
                                n_iter=n_iter, cv=kfold, scoring=metric, 
                                n_jobs=n_jobs, verbose=verbose)
    rs_clf.fit(X_train, y_train)
    print(f'Best score: {rs_clf.best_score_}')
    return rs_clf.best_estimator_


@run.command()
@click.argument('algorithm_acronym')
@click.argument('train_data_file')
@click.option('--algorithm_name', help='Name of algorithm to be trained', \
              default='', is_flag=False)
@click.option('--num_splits', help='Number of cross-validation splits', \
              default=3, is_flag=False)
@click.option('--num_iter', help='Number of iterations', \
              default=100, is_flag=False)
@click.option('--metric', help='Optimization metric', \
              default='accuracy', is_flag=False)
def train_model(algorithm_acronym, train_data_file, algorithm_name, num_splits, 
                num_iter, metric):
    # get data    
    with open(train_data_file, 'rb') as f:
        data = joblib.load(f)
    train_data = data['train_data']
    y_train = train_data['label'].values
    text_features = list(train_data.iloc[:,0].values)        
    extra_features = np.array(train_data.iloc[:,2:].values)
    X_train = np.concatenate((text_features, extra_features), axis=1)
     # define cross-validation
    kfold = KFold(n_splits=int(num_splits), shuffle=True, random_state=42)
    if algorithm_acronym == 'GB':
        param_grid = define_parameters_gb()
        best_model = do_hyperparametrization(GradientBoostingClassifier(), 
                                             param_grid, X_train, y_train, 
                                             int(num_iter), kfold, metric, -1, 1)
    else:
        raise Exception(f'Unknown algorithm acronym: {algorithm_acronym}')
    save_model(best_model, f'best_{algorithm_name}', algorithm_acronym, 
               metric, train_data_file)


if __name__ == "__main__":
    run()