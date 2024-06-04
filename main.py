import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#2)
df = pd.read_csv('https://raw.githubusercontent.com/marcelovca90-inatel/AG002/main/palmerpenguins.csv')

#3)
df.replace({'Biscoe':0, 'Dream':1, 'Torgersen':2, 'FEMALE':0, 'MALE':1, 'Adelie':0, 'Chinstrap':1, 'Gentoo':2}, inplace=True)

#4)
new_df = df.reindex(columns=['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species'])
print(new_df)

#5)
X = new_df.drop('species', axis=1)
y = new_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#7)
classifier = {
    "DT" : DecisionTreeClassifier()
}
for name, model in classifier.items():
    
    model.fit(X_train, y_train)
    
    # Scoring on SEEN data - effectively "useless"
    y_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred)
    
    # Scoring on UNSEEN data - important
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

#8) 
    report = classification_report(y_test, y_pred)
    print(report)
    
    print(f"{name:20s} accuracy\ttrain = {train_accuracy:.2%} \ttest = {test_accuracy:.2%}")

#9)
user_data = []
user_data.append(input("Enter island (Biscoe, Dream, Torgersen): "))
user_data.append(input("Enter sex (FEMALE, MALE): "))
user_data.append(float(input("Enter culmen length (mm): ")))
user_data.append(float(input("Enter culmen depth (mm): ")))
user_data.append(float(input("Enter flipper length (mm): ")))
user_data.append(float(input("Enter body mass (g): ")))

# Convert user data to dataframe
user_df = pd.DataFrame([user_data], columns=['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'])

# Preprocess user data
user_df.replace({'Biscoe':0, 'Dream':1, 'Torgersen':2, 'FEMALE':0, 'MALE':1}, inplace=True)

# Predict the species
user_pred = model.predict(user_df)

# Map the predicted label to the actual species
species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
predicted_species = species_mapping[user_pred[0]]

print(f"The predicted species for the given data is: {predicted_species}")
