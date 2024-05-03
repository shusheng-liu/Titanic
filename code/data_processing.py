import pandas as pd
import numpy as np

#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
#1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S

def read_data():
    
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    
    data = pd.read_csv("data/train.csv")
    train_label = data["Survived"].to_numpy()
    data.drop(['PassengerId','Survived','Name','Ticket', 'Cabin'], axis=1, inplace=True)
    for key, value in data.iterrows():
        temp = []
        temp.append(float(value.iat[0]))
        temp.append(0.0) if (value.iat[1] == "male") else temp.append(1.0)
        temp.append(0.0) if (np.isnan(value.iat[2])) else temp.append(float(value.iat[2]))
        temp.append(float(value.iat[3]))
        temp.append(float(value.iat[4]))
        if (value.iat[6] == "S"):
            temp.append(0.0)
        elif (value.iat[6] == "C"):
            temp.append(1.0)
        else:
            temp.append(2.0)
        train_data.append(temp)
    train_data = np.array(train_data)
  
    data = pd.read_csv(r"data/train.csv")
    test_label = data["Survived"].to_numpy()
    data.drop(['PassengerId','Survived','Name','Ticket', 'Cabin'], axis=1, inplace=True)
    for key, value in data.iterrows():
        temp = []
        temp.append(float(value.iat[0]))
        temp.append(0.0) if (value.iat[1] == "male") else temp.append(1.0)
        temp.append(0.0) if (np.isnan(value.iat[2])) else temp.append(float(value.iat[2]))
        temp.append(float(value.iat[3]))
        temp.append(float(value.iat[4]))
        if (value.iat[6] == "S"):
            temp.append(0.0)
        elif (value.iat[6] == "C"):
            temp.append(1.0)
        else:
            temp.append(2.0)
        test_data.append(temp)
    test_data = np.array(test_data)
    
    return([train_data, train_label, test_data, test_label])