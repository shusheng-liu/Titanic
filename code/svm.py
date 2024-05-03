import pandas as pd
from sklearn.svm import SVC
import numpy as np

def prepare_data(csv):
    data = []
    csv = csv.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
    for key, value in csv.iterrows():
        temp = []
        temp.append(np.float32(value.iat[0]))
        temp.append(np.float32(0)) if (value.iat[1] == "male") else temp.append(np.float32(1))
        temp.append(np.float32(0)) if (np.isnan(value.iat[2])) else temp.append(np.float32(value.iat[2]))
        temp.append(np.float32(value.iat[3]))
        temp.append(np.float32(value.iat[4]))
        if (value.iat[6] == "S"):
            temp.append(np.float32(0))
        elif (value.iat[6] == "C"):
            temp.append(np.float32(1))
        else:
            temp.append(np.float32(2))
        data.append(temp)
    return np.array(data)    
    
    
def main():
    # prepare data and train SVM model
    svm = SVC(kernel='linear')
    csv = pd.read_csv("data/train.csv", dtype={"Survived": np.float32})
    y_train = csv["Survived"].to_numpy()
    csv = csv.drop(['Survived'], axis=1)
    x_train = prepare_data(csv)
    svm.fit(x_train,y_train)
    print("Finished Training")
    
    # predict and output to csv
    csv = pd.read_csv("data/test.csv")
    id_test = csv['PassengerId'].to_numpy()
    x_test = prepare_data(csv)
    y_pred = svm.predict(x_test)
    y_pred = y_pred.astype(np.int64)
    dataset = pd.DataFrame({'PassengerId': id_test, 'Survived': y_pred})
    dataset.to_csv("output.csv", index = False)
    print("Outputting predictions.")
    
if __name__ == "__main__":
    main()