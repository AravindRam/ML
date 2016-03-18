import pandas as pd
import matplotlib.pyplot as plt

def plot_test_data(algoNum):
#Plot the bar chart for the predicted probabilities of Naive Bayes classifier
    if(algoNum == 1):
        df_test=pd.DataFrame.from_csv("decisionTree_test.csv",index_col=False,parse_dates=True)
    elif(algoNum == 2):
        df_test=pd.DataFrame.from_csv("adaBoost_test.csv",index_col=False,parse_dates=True)
    elif(algoNum == 3):
        df_test=pd.DataFrame.from_csv("naiveBayes_test.csv",index_col=False,parse_dates=True)
    elif(algoNum == 4):
        df_test=pd.DataFrame.from_csv("logisticRegression_test.csv",index_col=False,parse_dates=True)
    dist_count1 = df_test.groupby("Predicted").count()
    plt.figure()
    dist_count1.sort(columns="Id",ascending=1)["Id"].plot(kind="barh")
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    plt.tight_layout()
    if(algoNum == 1):
        plt.savefig('decisionTree_test.png') # Save the figure
    elif(algoNum == 2):
        plt.savefig('adaBoost_test.png') # Save the figure
    elif(algoNum == 3):
        plt.savefig('naiveBayes_test.png') # Save the figure
    elif(algoNum == 4):
        plt.savefig('logisticRegression_test.png') # Save the figure

    plt.show()

plot_test_data(1)   #For Decision Tree
plot_test_data(2)   #For Adaboost
plot_test_data(3)   #For Naive Bayes
plot_test_data(4)   #For Logistic Regression

