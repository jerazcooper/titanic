# import matplotlib
import numpy
import pandas

# from matplotlib import pyplot
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


"""
# setups for visualization

matplotlib.style.use("ggplot")

pandas.options.display.max_columns = 100
pandas.options.display.max_rows = 100


data = pandas.read_csv("train.csv")
data["Age"].fillna(data["Age"].median(), inplace=True)
"""

"""
# relation: sex to survival
survived_sex = data[data["Survived"] == 1]["Sex"].value_counts()
dead_sex = data[data["Survived"] == 0]["Sex"].value_counts()
dataframe = pandas.DataFrame([survived_sex, dead_sex])
dataframe.index = ["Survived", "Dead"]
dataframe.plot(kind="bar", stacked=True, figsize=(15, 8))
"""

"""
# relation: age to survival
figure = pyplot.figure(figsize=(15, 8))
pyplot.hist(
    [data[data["Survived"] == 1]["Age"],
     data[data["Survived"] == 0]["Age"]],
    stacked=True,
    color=["g", "r"],
    bins=30,
    label=["Survived", "Dead"],
)
pyplot.xlabel("Age")
pyplot.ylabel("Number of passengers")
pyplot.legend()
pyplot.show()
"""

"""
# relation: ticket price to survival
figure = pyplot.figure(figsize=(15, 8))
pyplot.hist(
    [data[data["Survived"] == 1]["Fare"],
     data[data["Survived"] == 0]["Fare"]],
    stacked=True,
    color=["g", "r"],
    bins=30,
    label=["Survived", "Dead"],
)
pyplot.xlabel("Fare")
pyplot.ylabel("Number of passengers")
pyplot.legend()
pyplot.show()
"""

"""
# relation: age and fare to survival
pyplot.figure(figsize=(15, 8))
axis_plot = pyplot.subplot()
axis_plot.scatter(
    data[data["Survived"] == 1]["Age"],
    data[data["Survived"] == 1]["Fare"],
    c="green",
    s=40,
)

axis_plot.scatter(
    data[data["Survived"] == 0]["Age"],
    data[data["Survived"] == 0]["Fare"],
    c="red",
    s=40,
)

axis_plot.set_xlabel("Age")
axis_plot.set_ylabel("Fare")
axis_plot.legend(("Survived", "Dead"), scatterpoints=1, loc="upper right", fontsize=15)
pyplot.show()
"""

"""
# relation: fare to passenger class
axis_plot = pyplot.subplot()
axis_plot.set_ylabel("Average Fare")
data.groupby("Pclass").mean()["Fare"].plot(kind="bar", figsize=(15, 8), ax=axis_plot)
pyplot.show()
"""

"""
# relation: embarkation site to survival
survived_embark = data[data["Survived"] == 1]["Embarked"].value_counts()
dead_embark = data[data["Survived"] == 0]["Embarked"].value_counts()
dataframe = pandas.DataFrame([survived_embark, dead_embark])
dataframe.index = ["Survived", "Dead"]
dataframe.plot(kind="bar", stacked=True, figsize=(15, 8))
pyplot.show()
"""


def status(feature):
    print("Processing {}: OK".format(feature))


def combined_data():
    training_data = pandas.read_csv("train.csv")
    test_data = pandas.read_csv("test.csv")

    # targets = training_data.Survived
    training_data.drop("Survived", axis=1, inplace=True)

    combined = training_data.append(test_data)
    combined.reset_index(inplace=True)
    combined.drop("index", axis=1, inplace=True)

    return combined


def extract_titles(data):
    data["Title"] = data["Name"].map(
        lambda name: name.split(",")[1].split(".")[0].strip()
    )

    title_dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty",
    }

    data["Title"] = data.Title.map(title_dictionary)
    return data


def fill_ages(row, grouped_by_median):
    return grouped_by_median.loc[row["Sex"], row["Pclass"], row["Title"]]["Age"]


def process_age(data):
    grouped_training_data = data.head(891).groupby(["Sex", "Pclass", "Title"])
    grouped_median_training_data = grouped_training_data.median()

    grouped_test_data = data.iloc[891:].groupby(["Sex", "Pclass", "Title"])
    grouped_median_test_data = grouped_test_data.median()

    data.head(891).Age = data.head(891).apply(
        lambda r: fill_ages(r, grouped_median_training_data)
        if numpy.isnan(r["Age"]) else r["Age"],
        axis=1,
    )

    data.iloc[891:].Age = data.iloc[891:].apply(
        lambda r: fill_ages(r, grouped_median_test_data)
        if numpy.isnan(r["Age"]) else r["Age"],
        axis=1,
    )

    return data


def process_names(data):
    data.drop("Name", axis=1, inplace=True)
    title_dummies = pandas.get_dummies(data["Title"], prefix="Title")
    data = pandas.concat([data, title_dummies], axis=1)

    data.drop("Title", axis=1, inplace=True)

    return data


def process_fares(data):
    data.head(891).Fare.fillna(data.head(891).Fare.mean(), inplace=True)
    data.iloc[891:].Fare.fillna(data.iloc[891:].Fare.mean(), inplace=True)
    return data


def process_embarked(data):
    # "S" is the most common
    data.head(891).Embarked.fillna("S", inplace=True)
    data.iloc[891:].Embarked.fillna("S", inplace=True)

    embarked_dummies = pandas.get_dummies(data["Embarked"], prefix="Embarked")
    data = pandas.concat([data, embarked_dummies], axis=1)
    data.drop("Embarked", axis=1, inplace=True)

    return data


def process_cabin(data):
    # "U" for unknown
    data.Cabin.fillna("U", inplace=True)

    # cabin name -> first letter of cabin name
    data["Cabin"] = data["Cabin"].map(lambda c: c[0])

    cabin_dummies = pandas.get_dummies(data["Cabin"], prefix="Cabin")
    data = pandas.concat([data, cabin_dummies], axis=1)
    data.drop("Cabin", axis=1, inplace=True)
    return data


def process_sex(data):
    data["Sex"] = data["Sex"].map({"male": 1, "female": 0})
    return data


def process_pclass(data):
    pclass_dummies = pandas.get_dummies(data["Pclass"], prefix="Pclass")
    data = pandas.concat([data, pclass_dummies], axis=1)
    data.drop("Pclass", axis=1, inplace=True)
    return data


def clean_ticket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = [p.strip() for p in ticket]
    ticket = [p for p in ticket if not p.isdigit()]
    # ticket = map(lambda t: t.strip(), ticket)
    # ticket = filter(lambda t: not t.isdigit(), ticket)

    return ticket[0] if ticket else "XXX"


def process_ticket(data):
    data["Ticket"] = data["Ticket"].map(clean_ticket)
    ticket_dummies = pandas.get_dummies(data["Ticket"], prefix="Ticket")
    data = pandas.concat([data, ticket_dummies], axis=1)
    data.drop("Ticket", inplace=True, axis=1)
    return data


def process_family(data):
    data["FamilySize"] = data["Parch"] + data["SibSp"] + 1
    data["Singleton"] = data["FamilySize"].map(lambda s: 1 if s == 1 else 0)
    data["SmallFamily"] = data["FamilySize"].map(lambda s: 1 if 2 <= s <= 4 else 0)
    data["LargeFamily"] = data["FamilySize"].map(lambda s: 1 if s >= 5 else 0)

    return data


def processed_data():
    combined = combined_data()

    combined = extract_titles(combined)

    combined = process_age(combined)
    combined = process_names(combined)
    combined = process_fares(combined)
    combined = process_embarked(combined)
    combined = process_cabin(combined)
    combined = process_sex(combined)
    combined = process_pclass(combined)
    combined = process_ticket(combined)
    combined = process_family(combined)

    return combined


def compute_score(classifier, X, y, scoring="accuracy"):
    xval = cross_val_score(classifier, X, y, cv=5, scoring=scoring)
    return numpy.mean(xval)


def get_train_test_targets(data):
    return data.head(891), data.iloc[891:], pandas.read_csv("train.csv").Survived


def get_model():
    training_data, test_data, targets = get_train_test_targets(processed_data())

    classifier = RandomForestClassifier(n_estimators=50, max_features="sqrt")
    classifier.fit(training_data, targets)

    features = pandas.DataFrame()
    features["feature"] = training_data.columns
    features["importance"] = classifier.feature_importances_
    features.sort_values(by=["importance"], ascending=True, inplace=True)
    features.set_index("feature", inplace=True)

    model = SelectFromModel(classifier, prefit=True)
    # reduced_training_data = model.transform(training_data)
    # reduced_test_data = model.transform(test_data)

    parameters = {
        "bootstrap": False,
        "min_samples_leaf": 3,
        "n_estimators": 50,
        "min_samples_split": 10,
        "max_features": "sqrt",
        "max_depth": 6,
    }

    model = RandomForestClassifier(**parameters)
    model.fit(training_data, targets)
    # compute_score(model, training_data, targets, scoring="accuracy")

    return model


def predict():
    _, test_data, _ = get_train_test_targets(processed_data())

    model = get_model()
    dataframe = pandas.DataFrame()
    aux = pandas.read_csv("test.csv")
    dataframe["PassengerId"] = aux["PassengerId"]

    dataframe["Survived"] = model.predict(test_data).astype(int)
    dataframe[["PassengerId", "Survived"]].to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    predict()
