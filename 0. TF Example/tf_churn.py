import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

if __name__ == '__main__':
    df = pd.read_csv('churn.csv')                                                   # (7044, 21)
    X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build the model (Two hidden, and fully-connected (Dense), layers):
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model:
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

    # Fit the model:
    model.fit(X_train, y_train, epochs=500, batch_size=32)

    # Predict:
    y_pred = model.predict(X_test)
    y_pred = [0 if y < 0.5 else 1 for y in y_pred]

    # Evaluate the model:
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    # Save the model:
    model.save('tf_churn_model')

    # Reload the model:
    model = load_model('tf_churn_model')
