import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LarsCV
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


white_wine = 0
clf = pd.read_csv('winequalityN.csv', header=0).fillna(0).values
for i in clf:
    if i[0] == 'white':
        i[0] = 0
        white_wine += 1
    else:
        i[0] = 1
x = clf[:, 0:12]
y = clf[:, 12]

for i in range(len(x[0])):
    x[:, i] = normalize([x[:, i]])

print('All:')
result = 0
for item in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # model = make_pipeline(StandardScaler(with_mean=False), LarsCV(normalize=False))
    model = LarsCV()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)

    success = 0
    for i in range(len(x_test)):
        if abs(y_test[i] - predict[i]) < 1:
            success += 1
    print(f'Accuracy {item + 1} split: {success / len(x_test) * 100:.4}%')
    result += success / len(x_test) * 100

print(f'Avg accuracy: {result / 10:.4}%\n')

x1 = clf[0:white_wine, 0:12]
y1 = clf[0:white_wine, 12]
for i in range(len (x1[0])):
    x1[:, i] = normalize([x1[:, i]])
print('White wine:')
result = 0
for item in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = LarsCV()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    success = 0
    for i in range(len(x_test)):
        if abs(y_test[i] - prediction[i]) < 1:
            success += 1
    print(f'Accuracy {item + 1} split: {success / len(x_test) * 100:.4}%')
    result += success / len(x_test) * 100

print(f'Avg accuracy: {result / 10:.4}%\n')

x2 = clf[white_wine:, 0:12]
y2 = clf[white_wine:, 12]
for i in range(len(x2[0])):
    x2[:, i] = normalize([x2[:, i]])
print('Red wine:')
result = 0
for item in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = LarsCV()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    success = 0
    for i in range(len(x_test)):
        if abs(y_test[i] - prediction[i]) < 1:
            success += 1
    print(f'Accuracy {item + 1} split: {success / len(x_test) * 100:.4}%')
    result += success / len(x_test) * 100

print(f'Avg accuracy: {result / 10:.4}%\n')
