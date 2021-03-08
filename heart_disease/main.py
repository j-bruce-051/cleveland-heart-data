import pandas as pd 
import matplotlib.pyplot as plt
import random
from Helpers import DataHelpers, Classifiers, HyperparameterTuners
import warnings 
warnings.filterwarnings("ignore")

#LOAD DATA
col_names = ['age', 'sex', 'cp', 'rest_bp', 'chol', 'fast_bs', 'rest_ecg', 'max_hr',
    'ex_ang', 'old_peak', 'slope', 'col_art', 'thal', 'label']
df = pd.read_csv("processed.cleveland.data", header = None)
df.columns = col_names
print(df.head())

# PRE PROCESSING
#get rid of the characters not allowed
df['col_art'], df['thal']= pd.to_numeric(df['col_art'], errors='coerce'), pd.to_numeric(df['thal'], errors='coerce')
row_num = len(df.index)
bad_row_count = sum(df.apply(lambda x: sum(x.isnull().values), axis = 1)>0) # count rows with a missing value
if bad_row_count < 0.05*row_num:
    df = df.dropna()
    df = df.reset_index()
else:
    # if too many rows contain NA, examine data and consider interpolating or inferring values if appropriate
    # deciding whether to drop also depends on usefulness of the missing fields
    print('more than 5 percent are bad rows, consider interpolating or inferring')
print(sum(df.apply(lambda x: sum(x.isnull().values), axis = 1)>0)) # verify no bad rows

# replace any value indicating severity with value just to indicate presence of disease for binary classification
df['label'] = df['label'].replace([2, 3, 4], 1) 


# are the labels approximately evenly distributed?
counts = df['label'].value_counts()
df['label'].hist() # 159 without, 137 with heart disease, so approx balanced
plt.show() #uncomment for hist 
# are the features? Yes

# catagorical data must be one hot encoded
df = DataHelpers.one_hot_encode(df, 'cp', ['typ_angina', 'atyp_angina', 'non_ang_pain', 'asymptomatic_cp'])
df = DataHelpers.one_hot_encode(df, 'thal', ['no_thal_defect', 'fixed_thal_defect', 'reversible_thal_defect'])
df = DataHelpers.one_hot_encode(df, 'slope', ['ST_upslope', 'ST_flat', 'ST_downslope']) # stress test ECG results slope direction
df = DataHelpers.one_hot_encode(df, 'rest_ecg', ['r_ecg_normal', 'r_ecg_ST_abnormal', 'r_ecg_hypertrophy']) # rest ecg

df = DataHelpers.scale_features(df) # scaling features

# find guesses of best features 
k_best_features = DataHelpers.select_k_best(df, k = 10) # function's guess at best features
df_k_best = df[k_best_features]
df_k_best['label'] = df['label'].copy()


#kendall_corr= df[df.columns[0:]].corr(method = "kendall")['label'][:] # correlations between features and label
kendall_corr = df[df.columns[0:]].corr(method = 'kendall')['label'][:]
s = abs(kendall_corr) > 0.3 # features with correlation with label over 0.3
kendall_best_features = list(s[s].index.values)
df_kendall_best = df[s[s].index.values]

# examine whether these are best features by predicting labels using full feature set and these feature sets
# a good best fetaure set would return the same or better accuracy 

# create model 
best_knn_params_dict = HyperparameterTuners.tune_KNN(df)
best_svc_params_dict = HyperparameterTuners.tune_SVC(df)
from numpy import array 
accuracy_knn_full, recall_knn_full = [],[]
accuracy_svc_full, recall_svc_full = [],[]
accuracy_knn_kendall, recall_knn_kendall = [],[]
accuracy_svc_kendall, recall_svc_kendall = [],[]
accuracy_knn_k_best, recall_knn_k_best = [],[]
accuracy_svc_k_best, recall_svc_k_best = [],[]

no_repeats = 31 # how many times to average the experiment over, 31 gives 30 runs

for i in range(0,no_repeats): 

    a,b = Classifiers.use_KNN(df, best_knn_params_dict)
    c,d = Classifiers.use_SVC(df, best_svc_params_dict)
    accuracy_knn_full.append(a), recall_knn_full.append(b), accuracy_svc_full.append(c), recall_svc_full.append(d)

    a,b = Classifiers.use_KNN(df_kendall_best, best_knn_params_dict)
    c,d = Classifiers.use_SVC(df_kendall_best, best_svc_params_dict)
    accuracy_knn_kendall.append(a), recall_knn_kendall.append(b), accuracy_svc_kendall.append(c), recall_svc_kendall.append(d) 

    a,b = Classifiers.use_KNN(df_k_best, best_knn_params_dict)
    c,d = Classifiers.use_SVC(df_k_best, best_svc_params_dict)
    accuracy_knn_k_best.append(a), recall_knn_k_best.append(b), accuracy_svc_k_best.append(c), recall_svc_k_best.append(d)


print('KNN accuracy:')
print("Full set : {}, Kendall : {}, k_best : {} ".format(DataHelpers.calc_mean_std(accuracy_knn_full), DataHelpers.calc_mean_std(accuracy_knn_kendall), DataHelpers.calc_mean_std(accuracy_knn_k_best)))
print('KNN recall:')
print("Full set : {}, Kendall : {}, k_best : {}".format(DataHelpers.calc_mean_std(recall_knn_full), DataHelpers.calc_mean_std(recall_knn_kendall), DataHelpers.calc_mean_std(recall_knn_k_best)))
print("SVC accuracy: ")
print("Full set : {}, Kendall : {}, k_best : {} ".format(DataHelpers.calc_mean_std(accuracy_svc_full), DataHelpers.calc_mean_std(accuracy_svc_kendall), DataHelpers.calc_mean_std(accuracy_svc_k_best)))
print('SVC recall:')
print("Full set : {}, Kendall : {}, k_best : {} ".format(DataHelpers.calc_mean_std(recall_svc_full), DataHelpers.calc_mean_std(recall_svc_kendall), DataHelpers.calc_mean_std(recall_svc_k_best)))



print("These features are common to two sets which perform as well as the full set: ")
print()
common_best_features = list(set(kendall_best_features).intersection(k_best_features))
print(common_best_features)


#let's just see if they perform well as a set themselves...
df_common_best = df[common_best_features]
df_common_best['label'] = df['label'].copy()

accuracy_knn_common_best, recall_knn_common_best = [],[]
accuracy_svc_common_best, recall_svc_common_best = [],[]
for i in range(0, no_repeats):
    a,b = Classifiers.use_KNN(df_common_best, best_knn_params_dict)
    c,d = Classifiers.use_SVC(df_common_best, best_svc_params_dict)
    accuracy_knn_common_best.append(a), recall_knn_common_best.append(b), accuracy_svc_common_best.append(c), recall_svc_common_best.append(d)
print("Using features common to both sets of good feature precictions: ")
print("KNN accuracy: {}, KNN recall: {}, SVC accuracy: {}, SVC recall: {}".format(DataHelpers.calc_mean_std(accuracy_knn_common_best),
    DataHelpers.calc_mean_std(recall_knn_common_best),DataHelpers.calc_mean_std(accuracy_svc_common_best),DataHelpers.calc_mean_std(recall_svc_common_best),))

# seems to be about the same? 

# one last thing, let's check it's worse with random features

feats = list(df.columns)
accuracy_knn_random, recall_knn_random = [],[]
accuracy_svc_random, recall_svc_random = [],[]

for i in range(0,no_repeats):

    random_features = random.sample(feats, k = 9) #sample without replacement each time, 9 is number in best 
    #print(random_features)
    df_random = df[random_features]
    try:
        print(df_random['label'].head(1)) # make sure we have label in here
    except:
        df_random['label'] = df['label'].copy()

    a,b = Classifiers.use_KNN(df_random, best_knn_params_dict)
    c,d = Classifiers.use_SVC(df_random, best_svc_params_dict)
    accuracy_knn_random.append(a), recall_knn_random.append(b), accuracy_svc_random.append(c), recall_svc_random.append(d)

print("Using random features: ")
print("KNN accuracy: {}, KNN recall: {}, SVC accuracy: {}, SVC recall: {}".format(DataHelpers.calc_mean_std(accuracy_knn_random),
    DataHelpers.calc_mean_std(recall_knn_random),DataHelpers.calc_mean_std(accuracy_svc_random),DataHelpers.calc_mean_std(recall_svc_random),))

# randomly selected features seem to do worse than the best sets, but not by much
# is that because 8/24 gives a good chance of selecting some of the best features? Do we only need a few of the best ones to predict well?
# I know I said the last part would be the last part but I just want to see something else
# What if I select the worst features, or select randomly from the best? How many good features do we need to get the same predictive power?
# And how would that change with more data (can't answer)?


# features with lowest kendall correlation
s = abs(kendall_corr) < 0.2 # thisis the 11 worst features
kendall_worst_features = list(s[s].index.values)
df_kendall_worst = df[s[s].index.values]
df_worst = df[kendall_worst_features]
df_worst['label'] = df['label'].copy()
accuracy_knn_worst, recall_knn_worst = [],[]
accuracy_svc_worst, recall_svc_worst = [],[]
for i in range(0,no_repeats):
    a,b = Classifiers.use_KNN(df_worst, best_knn_params_dict)
    c,d = Classifiers.use_SVC(df_worst, best_svc_params_dict)
    accuracy_knn_worst.append(a), recall_knn_worst.append(b), accuracy_svc_worst.append(c), recall_svc_worst.append(d)
print("Using worst features from Kendall: ")
print("KNN accuracy: {}, KNN recall: {}, SVC accuracy: {}, SVC recall: {}".format(DataHelpers.calc_mean_std(accuracy_knn_worst),
    DataHelpers.calc_mean_std(recall_knn_worst),DataHelpers.calc_mean_std(accuracy_svc_worst),DataHelpers.calc_mean_std(recall_svc_worst),))

# and randomly selecting from the best 
# if this works, it might be better than finding the very best from the perspective I'm interested in today, because I want to know how few fields this would work on
# and it would be even better if it runs on little data and it can be from a smaller range of options


accuracy_knn_random_best, recall_knn_random_best = [],[]
accuracy_svc_random_best, recall_svc_random_best = [],[]

for i in range(0,no_repeats):

    random_best_features = random.sample(common_best_features, k = 5) # half of the good features, we're pushing it a bit... 
    #print(random_features)
    df_random_best = df[random_best_features]
    df_random_best['label'] = df['label'].copy()

    a,b = Classifiers.use_KNN(df_random_best, best_knn_params_dict)
    c,d = Classifiers.use_SVC(df_random_best, best_svc_params_dict)
    accuracy_knn_random_best.append(a), recall_knn_random_best.append(b), accuracy_svc_random_best.append(c), recall_svc_random_best.append(d)

print("Using random best features: ")
print("KNN accuracy: {}, KNN recall: {}, SVC accuracy: {}, SVC recall: {}".format(DataHelpers.calc_mean_std(accuracy_knn_random_best),
    DataHelpers.calc_mean_std(recall_knn_random_best),DataHelpers.calc_mean_std(accuracy_svc_random_best),DataHelpers.calc_mean_std(recall_svc_random_best),))



# ok, using the 11 worst features performs very badly, and 5 of the best 9 features works a little worse than them all, but actually pretty well for only 5 features
# I could try and find out which of them are the best 
# tbh I am happier knowing that you can pick 5 of the top 9 and they would work OK
# Being able to run models as succesfully on not only reduced data but data from a choice of features and it working is very interesting IMO
# I wonder how spotty/missing/inaccurate real medical data is? Is it all missing/bad in the same way? Does what's missing depend on area/hospital/type of patient etc?
# Models for different subsets of good/passable features could exist for situations where different info is known? 
# That's probably not necessary in the UK, we probably have good data. Maybe places with less access to equipment though
# I wonder if I could pick 5 of the top, say, half of all the features? Would that work?
