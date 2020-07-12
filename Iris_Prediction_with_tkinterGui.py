import pandas as pd
import numpy as np
from tkinter import *
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"iris.csv")

root = Tk()
root.title("IRIS Flower Prediction")
root.geometry('500x400')

Label(root, text="Sepal_length", font="Times 15").grid(row=0, column=0, padx=50, pady=10)
Label(root, text="Sepal_width", font="Times 15").grid(row=1, column=0, pady=10)
Label(root, text="Petal_length", font="Times 15").grid(row=2, column=0, pady=10)
Label(root, text="Petal_width", font="Times 15").grid(row=3, column=0, pady=10)
Label(root, text="Prediction", font="Times 20").grid(row=5, column=0, pady=10)

textbox = Text(root, height=3, width=20, )
textbox.grid(row=5, column=1)

input_text = StringVar()
input_text1 = StringVar()
input_text2 = StringVar()
input_text3 = StringVar()
result = StringVar()

e1 = Entry(root, font=1, textvariable=input_text)
e1.grid(row=0, column=1)
e2 = Entry(root, font=1, textvariable=input_text1)
e2.grid(row=1, column=1)
e3 = Entry(root, font=1, textvariable=input_text2)
e3.grid(row=2, column=1)
e4 = Entry(root, font=1, textvariable=input_text3)
e4.grid(row=3, column=1)


def entryClear():
    input_text.set("")
    input_text1.set("")
    input_text2.set("")
    input_text3.set("")
    result.set("")
    textbox.delete(1.0, END)


def getpredict():
    lst = [float(e1.get()), float(e2.get()), float(e3.get()), float(e4.get())]
    eg = np.array(lst)
    eg = eg.reshape(1, -1)

    x = df.iloc[:, :4].values
    y = df.iloc[:, 4:5].values

    model = DecisionTreeClassifier()
    model.fit(x, y)

    predict = model.predict(eg)
    textbox.insert(END, predict)


Button(root, text='Clear', font=10, width=8, bg="#9B0000", fg="white", command=entryClear).grid(row=4, column=0,
                                                                                                pady=10)
Button(root, text='Predict', font=10, width=8, bg="#39FD03", fg="white", command=getpredict).grid(row=4, column=1,
                                                                                                  pady=10)

root.mainloop()