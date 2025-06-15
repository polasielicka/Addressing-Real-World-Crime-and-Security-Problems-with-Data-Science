from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

def random_search_xgboost(X_train, y_train,
                          n_iter: int = 150,
                          cv: int = 3,
                          random_state: int = 42):

    param_dist = {
        "n_estimators": randint(100, 600),
        "learning_rate": uniform(0.01, 0.29),
        "max_depth": randint(3, 10),               
        "min_child_weight": uniform(0.5, 9.5),     
        "subsample": uniform(0.5, 0.5),           
        "colsample_bytree": uniform(0.5, 0.5),     
        "gamma": uniform(0, 0.4),                  
        "reg_alpha": uniform(0, 1.0),          
        "reg_lambda": uniform(0, 2.0)              
    }

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",          # fast histogram algorithm (CPU)
        random_state=random_state,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )

    random_search.fit(X_train, y_train)

    print("\nBest hyper-parameters found by RandomizedSearchCV:")
    for k, v in random_search.best_params_.items():
        print(f"   • {k:<18}: {v}")

    best_model = random_search.best_estimator_

    return best_model, random_search


def grid_search_xgboost(X_train, y_train,
                        param_grid: dict,
                        cv: int = 3,
                        random_state: int = 42):
    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest hyper-parameters found by GridSearchCV:")
    for k, v in grid_search.best_params_.items():
        print(f"   • {k:<18}: {v}")

    best_model = grid_search.best_estimator_

    return best_model, grid_search

### LEFTOVER CODE FOR REFERENCE ###

### one set of hyperparameters
# model = XGBRegressor(
    #     n_estimators=200,
    #     learning_rate=0.05,
    #     max_depth=6,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42
    # )

### GridSearchCV example

# best_model, rs_obj = random_search_xgboost(X_train, y_train)
# model = best_model

# Train model (using GridsearchCV for hyperparameter tuning)
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [4, 6, 8],
#     'subsample': [0.7, 0.8, 1.0],
#     'colsample_bytree': [0.7, 0.8, 1.0]
# }

# base_model = XGBRegressor(random_state=42)
# grid_search = GridSearchCV(
#     estimator=base_model,
#     param_grid=param_grid,
#     cv=3,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1,
#     verbose=1
# )
# grid_search.fit(X_train, y_train)
# model = grid_search.best_estimator_
# print("Best parameters found:", grid_search.best_params_)
# model.fit(X_train, y_train)
################################################################