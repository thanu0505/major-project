import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.feature_selection import SelectKBest, chi2
from tkinter import Tk, Label, Button, Text, Scrollbar, END, filedialog

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def importdata():
    global balance_data
    balance_data = pd.read_csv("clean.txt")
    return balance_data

def splitdataset(balance_data):
    X = balance_data.values[:, 0:37]
    Y = balance_data.values[:, 38]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X, Y, X_train, X_test, y_train, y_test

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.insert(END, "Dataset loaded\n\n")

def preprocess():
    global labels, columns, filename
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", 
               "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", 
               "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
               "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
               "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", 
               "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
               "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
               "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    
    labels = {"normal": 0, "neptune": 1, "warezclient": 2, "ipsweep": 3, "portsweep": 4, "teardrop": 5, "nmap": 6, 
              "satan": 7, "smurf": 8, "pod": 9, "back": 10, "guess_passwd": 11, "ftp_write": 12, "multihop": 13, 
              "rootkit": 14, "buffer_overflow": 15, "imap": 16, "warezmaster": 17, "phf": 18, "land": 19, 
              "loadmodule": 20, "spy": 21, "perl": 22, "saint": 23, "mscan": 24, "apache2": 25, "snmpgetattack": 26, 
              "processtable": 27, "httptunnel": 28, "ps": 29, "snmpguess": 30, "mailbomb": 31, "named": 32, 
              "sendmail": 33, "xterm": 34, "worm": 35, "xlock": 36, "xsnoop": 37, "sqlattack": 38, "udpstorm": 39}

    balance_data = pd.read_csv(filename)
    dataset = ""
    index = 0
    cols = ""
    for index, row in balance_data.iterrows():
        for i in range(0, 42):
            if isfloat(row[i]):
                dataset += str(row[i]) + ","
            if index == 0:
                cols += columns[i] + ','
        dataset += str(labels.get(row[41])) + "\n"
        if index == 0:
            cols += 'Label'
        index = 1

    with open("clean.txt", "w") as f:
        f.write(cols + "\n" + dataset)
    
    text.insert(END, "Removed non-numeric characters from dataset and saved inside clean.txt file\n\n")
    text.insert(END, "Dataset Information\n\n")
    text.insert(END, dataset + "\n\n")

def generateModel():
    global data, X, Y, X_train, X_test, y_train, y_test
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    text.delete('1.0', END)
    text.insert(END, "Training model generated\n\n")

def prediction(X_test, model):
    y_pred = model.predict(X_test)
    for i in range(len(X_test)):
        print(f"X={X_test[i]}, Predicted={y_pred[i]}")
    return y_pred

def cal_accuracy(y_test, y_pred, details):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) * 100
    text.insert(END, details + "\n\n")
    text.insert(END, "Accuracy: " + str(accuracy) + "\n\n")
    text.insert(END, "Report: " + str(classification_report(y_test, y_pred)) + "\n")
    text.insert(END, "Confusion Matrix: " + str(cm) + "\n\n\n\n\n")
    return accuracy

def runSVM():
    global svm_acc, X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = SVC(C=2.0, gamma='scale', kernel='rbf', random_state=2)
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    svm_acc = cal_accuracy(y_test, prediction_data, "SVM Accuracy, Classification Report & Confusion Matrix")

def runRandomForest():
    global random_acc, X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    random_acc = cal_accuracy(y_test, prediction_data, "Random Forest Algorithm Accuracy, Classification Report & Confusion Matrix")

def runDNN():
    global dnn_acc, X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    srhl_tanh = MLPRandomLayer(n_hidden=8, activation_func='tanh')
    cls = GenELMClassifier(hidden_layer=srhl_tanh)
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    dnn_acc = cal_accuracy(y_test, prediction_data, "DNN Algorithm Accuracy, Classification Report & Confusion Matrix")

def graph():
    import matplotlib.pyplot as plt
    import numpy as np

    height = [svm_acc, random_acc, dnn_acc]
    bars = ('SVM Accuracy', 'Random Forest Accuracy', 'DNN Accuracy')
    y_pos = np.arange(len(bars))

    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

main = Tk()
main.title("Deep Learning")
main.geometry("1300x1200")

font1 = ('times', 12, 'bold')

title = Label(main, text='Deep Learning Approach for Intelligent Intrusion Detection System')
title.config(bg='brown', fg='white', font=('times', 16, 'bold'), height=3, width=120)
title.place(x=0, y=5)

upload_button = Button(main, text="Upload NSL KDD Dataset", command=upload)
upload_button.place(x=50, y=100)
upload_button.config(font=font1)

pathlabel = Label(main, bg='brown', fg='white', font=font1)
pathlabel.place(x=300, y=100)

preprocess_button = Button(main, text="Preprocess Dataset", command=preprocess)
preprocess_button.place(x=50, y=150)
preprocess_button.config(font=font1)

model_button = Button(main, text="Generate Training Model", command=generateModel)
model_button.place(x=330, y=150)
model_button.config(font=font1)

runsvm_button = Button(main, text="Run SVM Algorithm", command=runSVM)
runsvm_button.place(x=610, y=150)
runsvm_button.config(font=font1)

runrandomforest_button = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
runrandomforest_button.place(x=870, y=150)
runrandomforest_button.config(font=font1)

rundnn_button = Button(main, text="Run DNN Algorithm", command=runDNN)
rundnn_button.place(x=50, y=200)
rundnn_button.config(font=font1)

graph_button = Button(main, text="Accuracy Graph", command=graph)
graph_button.place(x=330, y=200)
graph_button.config(font=font1)

font2 = ('times', 12, 'bold')
text = Text(main, height=30, width=150, font=font2)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=250)
text.config(font=font1)

main.config(bg='brown')
main.mainloop()

