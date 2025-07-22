import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer

def move_pixel(digit, direction='left'):
    plane = digit.reshape(28, 28)
    moved = np.zeros_like(plane)

    for i in range(28):
        for j in range(28):
            try:
                match direction:
                    case 'left':
                        moved[i][j-1] = plane[i][j] 
                    case 'right':
                        moved[i][j+1] = plane[i][j]
                    case 'top':
                        moved[i-1][j] = plane[i][j] 
                    case 'down':
                        moved[i+1][j] = plane[i][j] 
            except IndexError:
                pass
    
    return moved.flatten()

def load_data():
    data = pd.read_csv('data/mnist_784.csv')

    y = data['class']
    X = data.drop(columns=['class'])

    return X.to_numpy(), y.to_numpy()

X, y = load_data()
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

def expansion_dataset(train_set, y):
    directions = ['left', 'right', 'down', 'top']
    result_set  = []
    y_set = []
    for i in range(len(train_set)):
        result_set.append(train_set[i])
        y_set.append(y[i])
        for opt in directions:
            new_copy = move_pixel(train_set[i], direction=opt)
            result_set.append(new_copy)
            y_set.append(y[i])
    
    return np.array(result_set), np.array(y_set)


# print(f'Size train_set: {len(X_train)}')
# X_extended_set, y_extended_set = expansion_dataset(X_train, y_train)

def save_extended_set(X, y):
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv('data/extended_mnist.csv', index=False)

# save_extended_set(X_extended_set, y_extended_set)
# print(f'Size extended_set: {len(X_extended_set)}, y = {len(y_extended_set)}')

def load_extended_set():
    data = pd.read_csv('data/extended_mnist.csv')
    X = data.drop(columns=['label']).to_numpy()
    y = data['label'].to_numpy()

    return X, y

def make_digit_plot(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')
    plt.savefig('R3-classification/plots/digit.png')

X, y = load_extended_set()

# n = 6
# make_digit_plot(X[n])
# print(f'{y[n]}')

#====================================
#  SEARCH BEST PARAMS FOR MODEL
#====================================
def save_model(model):
    joblib.dump(model, 'MODELS/mnist-784-extended.pkl')

def to_float64(X):
    return X.astype('float64')

model = Pipeline(steps=[
    ('to_float', FunctionTransformer(to_float64)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_jobs=-1))
])

model.fit(X, y)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"ACCURACY: {accuracy:.4f}")
print(f"PRECISION: {precision:.4f}")
print(f"RECALL: {recall:.4f}")
print(f"F1: {f1:.4f}")

save_model(model)