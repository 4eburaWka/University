import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=["RiskLevel"]).values  # Признаки
    y = data["RiskLevel"].factorize()[0]  # Преобразование RiskLevel в числовые метки
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.long)),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.long)),
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    loss_values = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_values.append(epoch_loss)

        if epoch % 5 == 0:
            print(f"Эпоха [{epoch}/{num_epochs}], Средняя потеря: {epoch_loss:.4f}")

    return loss_values


def evaluate_model(model, test_loader):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(y_batch.numpy())

    return y_true, y_pred


def calculate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred) * 100
    conf_matrix = confusion_matrix(y_true, y_pred)

    return f1, accuracy, conf_matrix


def plot_results(y_test, y_pred, num_classes):
    plt.figure(figsize=(10, 6))
    plt.hist(y_test, bins=num_classes, alpha=0.5, label='True', color='blue', edgecolor='black')
    plt.hist(y_pred, bins=num_classes, alpha=0.5, label='Predicted', color='orange', edgecolor='black')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.title('True vs Predicted Risk Levels')
    plt.legend()
    plt.show()


def plot_loss(loss_values, num_epochs):
    plt.plot(range(1, num_epochs + 1), loss_values, label='Ошибка при обучении')
    plt.xlabel('Эпохи')
    plt.ylabel('Ошибка')
    plt.title('График ошибки при обучении')
    plt.legend()
    plt.show()


def main():
    # Загрузка и предобработка данных
    X, y = load_data(r"IAD/lab3/Maternal Health Risk Data Set.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))
    model = SimpleNN(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00063)

    num_epochs = 500
    loss_values = train_model(model, train_loader, criterion, optimizer, num_epochs)

    y_true, y_pred = evaluate_model(model, test_loader)
    f1, accuracy, conf_matrix = calculate_metrics(y_true, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    plot_results(y_test, y_pred, num_classes)
    plot_loss(loss_values, num_epochs)


if __name__ == "__main__":
    main()
