from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np

def preprocess(df):#非常好的处理方式，值得好好学习
    df = df.copy()
    
    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
    def ticket_number(x):
        items = x.split(" ")
        if items[0] == "LINE":
            return 0
        return x.split(" ")[-1].strip(" ")
        
    def ticket_item(x):
        items = x.split(" ")
        if items[0] == "LINE":
            print("!")
            return "LINE"
        if len(items) == 1:
            return "NONE"
        return "!".join(items[0:-1]) #连接除了最后一串数字之前的所有部分，巧妙
    
    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)  
    df["Ticket_item"] = df["Ticket_item"].str.replace('[./]', '', regex=True) #去掉票据文本特征里的杂字符           
    return df

train_data = pd.read_csv(r"D:\Pythonworks\kaggle\datasets\3\titanic\train.csv")
test_data = pd.read_csv(r"D:\Pythonworks\kaggle\datasets\3\titanic\test.csv")

all_features = pd.concat((train_data.iloc[:, 1:], test_data.iloc[:, 1:]),ignore_index=True)
#all_features = all_features.drop(['Cabin'],axis=1)
all_features = preprocess(all_features)

number_features = all_features.dtypes[all_features.dtypes!='object'].index
number_features = number_features.drop('Survived') # 从索引中移除'SalePrice'
#print(number_features)
all_features['Fare'] = (all_features['Fare'] - all_features['Fare'].mean()) / all_features['Fare'].std()
all_features['Age'] = (all_features['Age'] - all_features['Age'].mean()) / all_features['Age'].std()
all_features = all_features.drop(['Ticket'],axis=1)
all_features['Ticket_number'] = all_features['Ticket_number'].astype(float)
all_features['Ticket_number'] = (all_features['Ticket_number'] - all_features['Ticket_number'].mean()) / all_features['Ticket_number'].std()
all_features['Fare'] = all_features['Fare'].fillna(all_features['Fare'].mean())
all_features['Age'] = all_features['Age'].fillna(all_features['Age'].mean())
all_features = pd.get_dummies(all_features, dummy_na=True) #dummy_na会把缺失值也单开一列，作为独立的特征（哑变量处理）

n_train = train_data.shape[0]
train,label = all_features[:n_train],'Survived'
predictor=TabularPredictor(label=label).fit(train_data=train,presets='best_quality')
res = test_data.iloc[:,0]
res = pd.concat((pd.DataFrame(res),pd.DataFrame(np.int64(predictor.predict(all_features[n_train:])))),axis=1)
res.rename(columns={0:"PassengerId",1:"Survived"},inplace=True)
res.to_csv("../datasets/3/autoresult.csv",index=False)
#ensemble (训练一百个模型，然后把分类结果加起来求平均，用阈值实现二分类)
#超参数调整：
# tuner = tfdf.tuner.RandomSearch(num_trials=1000)
# tuner.choice("min_examples", [2, 5, 7, 10])
# tuner.choice("categorical_algorithm", ["CART", "RANDOM"])

# local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
# local_search_space.choice("max_depth", [3, 4, 5, 6, 8])

# global_search_space = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
# global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])

# #tuner.choice("use_hessian_gain", [True, False])
# tuner.choice("shrinkage", [0.02, 0.05, 0.10, 0.15])
# tuner.choice("num_candidate_attributes_ratio", [0.2, 0.5, 0.9, 1.0])


# tuner.choice("split_axis", ["AXIS_ALIGNED"])
# oblique_space = tuner.choice("split_axis", ["SPARSE_OBLIQUE"], merge=True)
# oblique_space.choice("sparse_oblique_normalization",
#                      ["NONE", "STANDARD_DEVIATION", "MIN_MAX"])
# oblique_space.choice("sparse_oblique_weights", ["BINARY", "CONTINUOUS"])
# oblique_space.choice("sparse_oblique_num_projections_exponent", [1.0, 1.5])

# # Tune the model. Notice the `tuner=tuner`.
# tuned_model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
# tuned_model.fit(train_ds, verbose=0)

# tuned_self_evaluation = tuned_model.make_inspector().evaluation()
# print(f"Accuracy: {tuned_self_evaluation.accuracy} Loss:{tuned_self_evaluation.loss}")