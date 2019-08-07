import pandas as pd
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

#embedded method with decision tree

#Filter method
def getLinearCorrelation(df, columns, target, method):
    corr = df[columns].corr(method=method)[target].abs()
    print(len(corr))
    return corr.fillna(0)

def decisionTreeFeatureImportance(X, y, path):
    regr = DecisionTreeRegressor(max_depth=7)
    regr.fit(X, y)
    sorted_features = sorted(zip(X.columns, regr.feature_importances_),key = lambda t: t[1], reverse=True)
    dtree_df = pd.DataFrame.from_dict(sorted_features)
    dtree_df.columns = ["Features", "Decision Tree"]
    dtree_df.set_index("Features", inplace=True)
    
    # used to plot the decision tree using graphviz
    export_graphviz(regr, feature_names=X.columns.values, class_names=regr.classes_, out_file=path+os.path.sep+'tree.dot') 
    
    return dtree_df


def decisionTreeFeatureImportanceClassifier(X, y, path):
    regr = DecisionTreeClassifier(max_depth=7)
    regr.fit(X, y)
    sorted_features = sorted(zip(X.columns, regr.feature_importances_),key = lambda t: t[1], reverse=True)
    dtree_df = pd.DataFrame.from_dict(sorted_features)
    dtree_df.columns = ["Features", "Decision Tree"]
    dtree_df.set_index("Features", inplace=True)
    
    # used to plot the decision tree using graphviz
    export_graphviz(regr, feature_names=X.columns.values, class_names=regr.classes_, out_file=path+os.path.sep+'tree.dot') 
    
    return dtree_df


#embedded method with LASSO


def LassoFeatureImportance(X, y, training_columns):
    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = LassoCV(cv=5)

    clf.fit(X,y)
    
    result_dict = {}
    for name, value in sorted(zip(training_columns , clf.coef_)):
        if value>0:
            result_dict[name]=value

    return pd.DataFrame.from_dict(result_dict, orient='index', columns=['LASSO coefficient'])

#embedded method with Xgboost


def xgboostFeatureImportance(X, y):
    model=xgb.XGBRegressor()
    model.fit(X, y)
    sorted_features = model.get_booster().get_score(importance_type='weight')
    xgboost_df = pd.DataFrame.from_dict(sorted_features, orient='index')
    xgboost_df.columns = ["XGBoost"]
    
#     xgb.plot_importance(model, max_num_features=10)
#     pyplot.show()
#     sorted(sorted_features.items(), key=lambda kv: kv[1], reverse=True)[:10]
    
    return xgboost_df    


def xgboostFeatureImportanceClassifier(X, y):
    model=xgb.XGBClassifier()
    model.fit(X, y)
    sorted_features = model.get_booster().get_score(importance_type='weight')
    xgboost_df = pd.DataFrame.from_dict(sorted_features, orient='index')
    xgboost_df.columns = ["XGBoost"]
    
    # xgb.plot_importance(model, max_num_features=10)
    # pyplot.show()
    # sorted(sorted_features.items(), key=lambda kv: kv[1], reverse=True)[:10]
    
    return xgboost_df  



def getTopFeatures(df_features, nb_features = 5):
    selected_features = []
    
    if isinstance(df_features, pd.DataFrame):
        for col in df_features.columns:
            if "coefficient" not in col:
                temp_list=list(df_features[col].sort_values().index[:nb_features].values)
                selected_features.append(temp_list)
    else: #it is a serie
        temp_list=list(df_features.sort_values().index[:nb_features].values)
        selected_features.append(temp_list)
            
    selected_features=[item for sublist in selected_features for item in sublist] #flatten the list
    selected_features=set(selected_features) #remove duplicates
        
    return list(selected_features)

#Remove the variable with low correlation / low information
def getInformationVariable(df, index_df, threshold=0.7, replace_col_name = False):
    correlation_matrix = df[index_df].corr()
    if replace_col_name:
        feature_names = ["feature "+str(x) for x in range(1, correlation_matrix.shape[1]+1, 1)]
        correlation_matrix.columns=feature_names
        correlation_matrix.set_index(pd.Index(feature_names), inplace=True)
        # correlation_matrix.to_csv(path_save+"correlation_matrix_pearson.csv")
    
    information_variables = []
        
    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if col!=row:
                score = correlation_matrix.loc[row, col]
                if score>threshold:
                     information_variables.append(col)
    information_variables = set( information_variables)
    return  information_variables

def formatting(list_values):
    list_values = [x.replace(' ', '_') for x in list_values]
    list_values = [x.replace('[', '(') for x in list_values]
    list_values = [x.replace(']', ')') for x in list_values]
    list_values = [x.replace('<', 'inf. to ') for x in list_values]
    list_values = [x.replace('>', 'sup. to ') for x in list_values]
    return list_values

def remove_formatting(list_values):
    list_values = [x.replace('_', ' ') for x in list_values]
    list_values = [x.replace('(', '[') for x in list_values]
    list_values = [x.replace(')', ']') for x in list_values]
    list_values = [x.replace('inf. to ', '<') for x in list_values]
    list_values = [x.replace('sup. to ', '>') for x in list_values]
    return list_values


def displayFeatureImportance(df_features):
    fig, ax= plt.subplots(figsize=(6, 10))
    df_features["Average ranking"] = df_features[["Pearson", "Spearman", "LASSO ranking", "XGBoost"]].mean(axis=1)
    df_features.sort_values("Average ranking")[["Pearson", "Spearman", "LASSO ranking", "XGBoost"]].head(15).plot.barh(ax=ax)
    ax.set_xlabel("Average ranking")
    return fig

def getFeatureImportance(df, target_col, training_columns):


    training_columns = df[training_columns].select_dtypes(include='number').columns

    #Remove any columns that are not numbers
    df.columns = formatting(df.columns)
    training_columns = formatting(training_columns)
   
    target_col = formatting([target_col])[0]

    training_columns = list(training_columns)
    result = pd.DataFrame(index = training_columns)

    ## Pearson correlation

    result["Pearson's coefficient"] = getLinearCorrelation(df, training_columns, target_col, method='pearson').values
    result["Pearson"] = result["Pearson's coefficient"].rank(ascending=False).values
    
   
    ## Spearman correlation
    result["Spearman's coefficient"] = getLinearCorrelation(df, training_columns, target_col, method='spearman').values
    result["Spearman"]=result["Spearman's coefficient"].rank(ascending=False).values

    ## Preparing data for machine learning algorithm
    X = df[training_columns].drop(target_col, axis=1).fillna(0)
    y = df[target_col]
#     X.columns=X.columns.str.replace(' ', '_') # Remove space in columns' name as it produces error when ploting the tree

    ### Using a decision tree
#     dtree_results = decisionTreeFeatureImportance(X, y)
#     result=pd.merge(result, dtree_results, left_index=True, right_index=True, how="left")
#     result["Decision Tree"].fillna(0, inplace=True)
#     result["Decision Tree"]=result["Decision Tree"].rank(ascending=False)
    
    ##Using LASSO
    lasso_results = LassoFeatureImportance(X, y, training_columns)
    result=pd.merge(result, lasso_results, left_index=True, right_index=True, how="left")
    result["LASSO coefficient"].fillna(0, inplace=True)
    result["LASSO ranking"]=result["LASSO coefficient"].rank(ascending=False).values

    ### using Xgboost
    xgboost_results = xgboostFeatureImportance(X, y)
    result=pd.merge(result, xgboost_results, left_index=True, right_index=True, how="left")
    result["XGBoost"].fillna(0, inplace=True)
    result["XGBoost"]=result["XGBoost"].rank(ascending=False)

    df.columns = remove_formatting(df.columns)
    result.index = remove_formatting(result.index)
    return result