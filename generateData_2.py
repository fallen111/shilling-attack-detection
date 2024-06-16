import os
from averageAttack import AverageAttack
from bandwagonAttack import BandWagonAttack
from randomAttack import RandomAttack
from hybridAttack import HybridAttack
from RR_Attack import RR_Attack 
from getMetrics import calcAndSaveMetrics
from generateConfigs import generateConfigFiles
import pickle 
import pandas as pd 
def generateData():
    origdf = pd.read_csv("dataset\\MovieLense\\ratings_no_timestamp.txt", sep=' ', engine='python', names=['userid', 'movie_id', 'rating'])
    print (f"finished loading the original dataset")
    generateConfigFiles()
    config_directory = "./config"
    output_directory = "./output"
    # Read the ratings_no_timestamp.txt file and create a dictionary of ratings
    # Iterate through config files in the config directory
    for config_file in os.listdir(config_directory):
        if config_file.endswith(".txt"):
            config_path = os.path.join(config_directory, config_file)
            print (f"start creating config file ** {config_file} **")
            attack_type = config_file.split("_")[0]  # Extracting attack type from file name
            attack_size = int(config_file.split("_")[-1].split(".")[0])  # Extracting attack size from file name
            if attack_type == "Average":
                attack = AverageAttack(config_path)
            elif attack_type == "BandWagon":
                attack = BandWagonAttack(config_path)
            elif attack_type == "Hybrid":
                attack = HybridAttack(config_path)
            elif attack_type == "Random":
                attack = RandomAttack(config_path)
            elif attack_type == "RR":
                attack = RR_Attack(config_path)
            else:
                print(f"Unknown attack type: {attack_type}")
                continue
            attack.insertSpam()
            # attack.farmLink()
            attack.generateLabels( "labels.txt")
            attack.generateProfiles( "profiles.txt")
            outputDir= parse_returnOutput_configFile(config_path)
            print (f"finished with injecting attack ###  {attack_type} _ {attack_size}###")
            df = pd.read_csv(os.path.join(outputDir ,'profiles.txt'), sep=' ', engine='python', names=['userid', 'movie_id', 'rating'])
            df = calcAndSaveMetrics(df)   
            print (f"finished calculating metics for  attack ###  {attack_type} _ {attack_size} ###")   
            # with open(os.path.join(output_directory,f'df_metrics_{attack_type}_{attack_size}.pickle', 'wb')) as f:
            #     pickle.dump(df, f)
            # Save the DataFrame with metrics to a CSV file
            df = compare_with_orig_label(df, origdf)
            df.to_csv(os.path.join(outputDir,f'df_metrics_{attack_type}_{attack_size}.csv'), index=False)
            print (f"successfully saved  attack ###  {attack_type} _ {attack_size} ### to CSV file ")

def compare_with_orig_label(df, origdf):
    # Read attack labels file
    # Merge df with attack labels
    merged_df = pd.merge(df, origdf, on=['userid', 'movie_id', 'rating'], how='left', indicator=True)
    # Label rows as 0 if they are in attack labels, else label them as 1
    merged_df['label'] = merged_df['_merge'].apply(lambda x: 1 if x == 'both' else 0)
    # Drop unnecessary columns
    # merged_df.drop(['rating_y', '_merge'], axis=1, inplace=True)
    merged_df.drop([ '_merge'], axis=1, inplace=True)
    # merged_df.rename(columns={'rating_x': 'rating'}, inplace=True)
    return merged_df

def parse_returnOutput_configFile(config_file):
    output_dir = None
    # Read the config file
    with open(config_file, 'r') as f:
        lines = f.readlines()
    # Iterate through each line in the config file
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        # Split the line into key and value based on '=' delimiter
        key_value = line.split('=')
        # Check if the line contains the outputDir key
        if key_value[0].strip() == 'outputDir':
            output_dir = key_value[1].strip()  # Extract the output directory address
            break  # No need to continue looping
    if not output_dir:
        raise Exception("no output found")
    return output_dir


generateData()