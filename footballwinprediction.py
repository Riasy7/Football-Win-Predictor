"""
Football Match Predictior
Made using Python, scikit-learn and pandas
Algorithm used: Random Forest Classifier
Accuracy improvement: 47% to 62.5%
-----------------------------------------------------
Written by Ahmad Saadawi
Github: https://github.com/Riasy7
repo: https://github.com/Riasy7/Football-Win-Predictor
-----------------------------------------------------
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score


matches = pd.read_csv("matches.csv", index_col=0) # load the data

# load some data and process and understand it (used print statements that were removed after understanding the data and making the necessary changes)
matches.head()
matches["team"].value_counts()
matches[matches["team"] == "Liverpool"]
matches["round"].value_counts()

# Preprocess the data
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opponent_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype(int)


rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1) # initialize rf model

# Inportant: Split the data into train and test sets
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']


predictors = ["venue_code", "opponent_code", "hour", "day_code"] # predictors for the model


rf.fit(train[predictors], train["target"]) # train the model


preds = rf.predict(test[predictors]) # make predictions


acc = accuracy_score(test["target"], preds) # calc accuracy score
acc # print accuracy score (prints 0.6123188405797102, 61% accuracy)

combined = pd.DataFrame(dict(actual=test["target"], prediction=preds)) # combine actual and predicted values


pd.crosstab(index=combined["actual"], columns=combined["prediction"]) # create matrix


precision_score(test["target"], preds) # calc precision score


grouped_matches = matches.groupby("team") # group by team
group = grouped_matches.get_group("Manchester City")

# func for calculating rolling averages
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"] # original cols
new_cols = [f"{c}_rolling" for c in cols] # new cols for rolling averages


rolling_averages(group, cols, new_cols) # call func on group


matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols)) # apply func to data

 
matches_rolling = matches_rolling.droplevel('team') # reset index and drop team column
matches_rolling.index = range(matches_rolling.shape[0])

# func for making new predictions with rolling averages
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision


combined, precision = make_predictions(matches_rolling, predictors + new_cols) # create prediction by calling func
# precision is now 0.625, 62.5% accuracy, a lot better than before

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True) # merge

# dictionary for mapping team names
class MissingDict(dict):
    __missing__ = lambda self, key: key

# mapping for team names
map_values = {
    "Birghton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)


combined["new_team"] = combined["team"].map(mapping) # apply the mapping to the data


merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"]) # merge the data


merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts() # count the values
