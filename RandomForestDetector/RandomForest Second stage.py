import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib
import pickle 
# Define the input folder path
def get_csv_files(directory):
    # csv_files = glob.glob(directory + '/*.csv')
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def extract_attack_info(filename):
    name_without_extension = os.path.splitext(filename)[0]
    parts = name_without_extension.split('_')
    return parts[2], int(float(parts[3]))

output_directory = "output_new"
csv_files = get_csv_files(output_directory)
print(csv_files)
# List of input files to process
# Different hyperparameter initializations for the Random Forest model
rf_params_list = [
    # {'n_estimators': 100, 'max_depth': None, 'criterion':"entropy"},
    # {'n_estimators': 100, 'max_depth': None,'criterion':"gini"}, 
    {'n_estimators': 500, 'max_depth': 70 ,'criterion':"entropy"},
    {'n_estimators': 500, 'max_depth': 70 ,'criterion':"gini"},
    # {'n_estimators': 100, 'max_depth': 70,'criterion':"entropy"},
    # {'n_estimators': 100, 'max_depth': 70,'criterion':"gini"},
    {'n_estimators': 500, 'max_depth': None,'criterion':"entropy"},
    {'n_estimators': 500, 'max_depth': None,'criterion':"gini"}
]
# Lists to store results
results = []
df_train = pd.read_csv(r"output\Hybrid_Attack_attackSize_100\df_metrics_Hybrid_100.csv", index_col=False)
df = df_train.reset_index(drop=True)
# Separate features and labels
X_train = df_train.drop(["label"], axis=1)
y_train = df_train["label"]
for i,rf_params in  enumerate(rf_params_list):
    print (f"started tarining on {rf_params}")
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    # Process each input file and parameter initialization
    for csv_file in csv_files:
        input_folder = os.path.split(csv_file)[0]
        df_test = pd.read_csv(csv_file, index_col=False)
        df_test = df_test.reset_index(drop=True)
        # Separate features and labels
        X_test = df_test.drop(["label"], axis=1)
        y_test = df_test["label"]
        attackType, attackSize = extract_attack_info(os.path.split(csv_file)[1])
        print("... Initializing training stage ")
        # Train a Random Forest model

        # Predict on the test set
        y_pred = model.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Classification report
        class_rep = classification_report(y_test, y_pred)
        input_file = f"SecondStage_{attackType}_{attackSize}_{rf_params['n_estimators']}_{rf_params['max_depth']}_{rf_params['criterion']}"
        results.append({
            "input_file": f"SecondStage_attackType_{attackType}_attackSize_{attackSize}_nEstimator_{rf_params['n_estimators']}_max_depth_{rf_params['max_depth']}_criterion_{rf_params['criterion']}",
            "rf_params": rf_params,
            "accuracy": accuracy
            # ,"confusion_matrix": cm,
            # "classification_report": class_rep,
            # "model": model
        })
        model_filename = os.path.join(input_folder, f"RF_{input_file}.pkl")
        joblib.dump(model, model_filename)
        # Save the confusion matrix and classification report
        cm_filename = os.path.join(input_folder, f"confusion_matrix_{input_file}.txt")
        with open(cm_filename, "w") as cm_file:
            cm_file.write(np.array2string(cm, separator=", "))
        class_rep_filename = os.path.join(input_folder, f"classification_report_{input_file}.txt")
        with open(class_rep_filename, "w") as class_rep_file:
            class_rep_file.write(class_rep)
            
        print(f"Input File: {input_file}")
        print(f"RF Params: {rf_params}")
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(class_rep)
        print("=" * 50)
        with open('overall_results.pkl', 'wb') as f:
            pickle.dump(results, f)

plt.figure(figsize=(15, 10))
for result in results:
    input_file = result["input_file"]
    rf_params = result["rf_params"]
    accuracy = result["accuracy"]
    plt.bar(f"{input_file}\n{rf_params}", accuracy)
plt.xlabel("Input Files and RF Params")
plt.ylabel("Accuracy")
plt.title("Accuracy of Random Forest Models with Different Parameters")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(input_folder, "accuracy_plot.png"))
print("Process completed.")

# def extract_attack_info(path):
#     # Split the path into directory and filename
#     directory, filename = os.path.split(path)
    
#     # Extract attack name and attack size from the filename
#     match = re.match(r'.*?_(.*?)_attackSize_(\d+\.\d+).*csv', filename)
#     if match:
#         attack_name = match.group(1)
#         attack_size = int(match.group(2))
#         return attack_name, attack_size
#     else:
#         return None, None




# for result in results:
#     input_file = result["input_file"]
#     rf_params = result["rf_params"]
#     accuracy = result["accuracy"]
#     cm = result["confusion_matrix"]
#     class_rep = result["classification_report"]
#     model = result["model"]
#     # Print results
