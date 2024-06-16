import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle 

#•	درجه توافق وزنی
def weighted_agreement_degree(df):
    # Calculate average ratings for each movie
    movie_avg_ratings = df.groupby('movie_id')['rating'].mean()
    
    # Calculate total number of ratings for each movie
    movie_total_ratings = df.groupby('movie_id')['rating'].count()
    
    # Merge average ratings and total ratings back to the original dataframe
    df = df.merge(movie_avg_ratings, on='movie_id', suffixes=('', '_avg'))
    df = df.merge(movie_total_ratings, on='movie_id', suffixes=('', '_total'))
    
    # Calculate weighted agreement degree for each rating
    df['WA'] = abs(df['rating'] - df['rating_avg']) / (df['rating_total'] ** 2)
    # Calculate weighted agreement degree for each user
    weighted_agreement_degree_u = df.groupby('userid')['WA'].sum()
    # wad_per_user = data.groupby('user_id').apply(lambda x: np.sum(x['rating_diff'] / x.shape[0]**2))
    return weighted_agreement_degree_u

#•	انحراف وزنی از میانگین توافق
def weighted_deviation_from_mean_agreement(df):
    # Calculate average ratings for each movie
    movie_avg_ratings = df.groupby('movie_id')['rating'].mean()
    
    # Calculate total number of ratings for each movie
    movie_total_ratings = df.groupby('movie_id')['rating'].count()
    
    # Merge average ratings and total ratings back to the original dataframe
    df = df.merge(movie_avg_ratings, on='movie_id', suffixes=('', '_avg'))
    df = df.merge(movie_total_ratings, on='movie_id', suffixes=('', '_total'))
    
    # Calculate weighted deviation from mean agreement for each rating
    df['WD'] = abs(df['rating'] - df['rating_avg']) / (df['rating_total'] ** 2)
    # Calculate WDMA_u for each user
    WDMA_u = df.groupby('userid')['WD'].sum() / df.groupby('userid').size()
    WDMA_u = WDMA_u.rename('WD')
    return WDMA_u



#•	درجه تشابه با همسایه های برتر
def similarity_with_top_neighbors_degree(df, k=5):
    # Calculate similarity matrix between users using cosine similarity
    user_similarity_matrix = cosine_similarity(df.pivot(index='userid', columns='movie_id', values='rating').fillna(0))
    
    # Convert similarity matrix to DataFrame
    user_similarity_df = pd.DataFrame(user_similarity_matrix, index=df['userid'].unique(), columns=df['userid'].unique())
    
    # Exclude self-similarity and sort neighbors by similarity
    user_similarity_df.values[[range(len(user_similarity_df))]*2] = 0
    top_neighbors = user_similarity_df.apply(lambda x: x.nlargest(k).index.tolist(), axis=1)
    
    # Calculate DegSim_u for each user
    DegSim_u = {}
    for user, neighbors in top_neighbors.items():
        similarity_sum = user_similarity_df.loc[user, neighbors].sum()
        DegSim_u[user] = similarity_sum / len(neighbors)
    
    return DegSim_u

#  نوان DegSim' 
def DegSim_prime(df, k=5):
    # Calculate DegSim_u for each user
    DegSim_u = similarity_with_top_neighbors_degree(df, k)
    
    # Calculate average DegSim_u
    avg_DegSim_u = sum(DegSim_u.values()) / len(DegSim_u)
    
    # Calculate DegSim' metric
    DegSim_prime = sum(abs(DegSim_u[user] - avg_DegSim_u) for user in DegSim_u)
    
    return DegSim_prime

#•	انحراف استاندارد در امتیازدهی کاربران
def standard_deviation_user_ratings(df):
    # Calculate mean ratings for each user
    user_mean_ratings = df.groupby('userid')['rating'].mean()
    # Merge mean ratings back to the original dataframe
    df = df.merge(user_mean_ratings, on='userid', suffixes=('', '_mean'))
    # Calculate squared deviations from mean
    df['SD'] = (df['rating'] - df['rating_mean']) ** 2
    # Calculate standard deviation for each user
    user_std_dev = df.groupby('userid')['SD'].sum() / (df.groupby('userid')['rating'].count() - 1)
    user_std_dev = np.sqrt(user_std_dev)
    user_std_dev = user_std_dev.rename ('SD')
    return user_std_dev


#•	درجه عدم توافق با سایر کاربران
def degree_of_disagreement(df):
    # Calculate average ratings for each item
    item_avg_ratings = df.groupby('movie_id')['rating'].mean()
    # Merge average ratings back to the original dataframe
    df = df.merge(item_avg_ratings, on='movie_id', suffixes=('', '_avg'))
    # Calculate absolute differences between user ratings and item average ratings
    df['abs_rating_diff'] = abs(df['rating'] - df['rating_avg'])
    # Calculate degree of disagreement for each user
    user_dd = df.groupby('userid')['abs_rating_diff'].sum() / df.groupby('userid')['rating'].count()
    user_dd = user_dd.rename('DD')
    return user_dd


#•	واریانس طول

def length_variance(df):
    # Calculate the number of ratings for each user
    user_rating_counts = df.groupby('userid')['rating'].count()
    # Calculate the average profile length
    average_profile_length = user_rating_counts.mean()
    # Calculate the numerator
    numerator = abs(user_rating_counts - average_profile_length)
    # Calculate the denominator
    denominator = (user_rating_counts - average_profile_length).pow(2).sum()
    # Calculate the length variance for each user
    length_variance = numerator / denominator
    length_variance = length_variance.rename('LV')
    return length_variance

def Hv_u(df):
    # Compute the overall average rating
    overall_avg_rating = df['rating'].mean()

    # Compute the average rating of each user
    user_avg_rating = df.groupby('userid')['rating'].mean()
    # Merge user average rating with the original dataframe
    df = pd.merge(df, user_avg_rating, left_on='userid', right_index=True, suffixes=('', '_user_avg'))

    user_avg_rating = df.groupby('movie_id')['rating'].mean()
    # Merge user average rating with the original dataframe
    df = pd.merge(df, user_avg_rating, left_on='movie_id', right_index=True, suffixes=('', '_movie_avg'))

    # Merge overall average rating with the original dataframe
    df['overall_avg_rating'] = overall_avg_rating

    def calculate_Hv_u(group):
        numerator = np.sum((group['rating'] - group['rating_user_avg'] - group['rating_movie_avg'] + group['overall_avg_rating'])**2)
        denominator = np.sum((group['rating'] - group['overall_avg_rating'])**2)
        return  numerator/ denominator
    Hv_u = df.groupby('userid').apply(calculate_Hv_u)
    Hv_u = Hv_u.rename ('Hv')
    return Hv_u
    # temp  = df.groupby('userid').apply(calculate_Hv_u)
    # return temp

# dataset_path = "dataset/MovieLense/ratings_no_timestamp.txt"
# df = pd.read_csv(dataset_path, sep=' ', engine='python', names=['userid', 'movie_id', 'rating'])
# print ("finished loading the")
# FileIO.loadDataSet(config, config['ratings'])
def calcAndSaveMetrics(df ):
    temp1 = weighted_agreement_degree(df)#WA_
    temp2 = weighted_deviation_from_mean_agreement(df)#WD_
    temp3 = standard_deviation_user_ratings(df)#SD_
    temp4 = degree_of_disagreement(df)#DD_
    temp5 = length_variance(df)#LV_
    temp6 = Hv_u(df)#Hv_
    df  = pd.merge(df, temp1, left_on='userid', right_index=True, suffixes=('','WA'))
    df  = pd.merge(df, temp2, left_on='userid', right_index=True, suffixes=('','WD'))
    df  = pd.merge(df, temp3, left_on='userid', right_index=True, suffixes=('','SD'))
    df  = pd.merge(df, temp4, left_on='userid', right_index=True, suffixes=('','DD'))
    df  = pd.merge(df, temp5, left_on='userid', right_index=True, suffixes=('','LV'))
    df  = pd.merge(df, temp6, left_on='userid', right_index=True, suffixes=('','Hv'))
    return df 
    # Save the merged dataframe to a file
    # df.to_csv('merged_dataframe_with_metrics.csv', index=False)
    # with open('merged_dataframe_with_metrics.pickle', 'wb') as f:
    #     pickle.dump(df, f)

# temp = similarity_with_top_neighbors_degree(df)#has problem
# temp = DegSim_prime(df)#has problem

# df = df.merge(temp1.add_prefix('WA_'), on='userid')
# df = df.merge(temp2.add_prefix('WD_'), on='userid')
# df = df.merge(temp3.add_prefix('SD_'), on='userid')
# df = df.merge(temp4.add_prefix('DD_'), on='userid')
# df = df.merge(temp5.add_prefix('LV_'), on='userid')
# df = df.merge(temp6.add_prefix('Hv_'), on='userid')
