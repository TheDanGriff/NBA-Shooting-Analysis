import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_shot_prediction_model(data_file):
    # Load synchronized data
    data = pd.read_csv(data_file)
    
    # Define features and labels
    X = data[['elbow_angle', 'wrist_angle', 'ball_x', 'ball_y']]
    y = data['shot_success']  # Assuming you have shot success labels (1 for made, 0 for missed)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    train_shot_prediction_model('merged_shot_data.csv')

