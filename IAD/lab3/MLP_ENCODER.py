import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = torch.relu(self.decoder(encoded))
        return decoded, encoded


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, pretrained_weights=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, num_classes)

        if pretrained_weights is not None:
            self._initialize_weights(pretrained_weights)

    def _initialize_weights(self, pretrained_weights):
        self.fc1.weight.data = pretrained_weights[0].weight.data
        self.fc2.weight.data = pretrained_weights[1].weight.data
        self.fc3.weight.data = pretrained_weights[2].weight.data
        self.fc4.weight.data = pretrained_weights[3].weight.data

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def pretrain_autoencoders(X_train_tensor, input_size, hidden_sizes):
    pretrained_weights = []
    data = X_train_tensor.clone()

    for idx, hidden_size in enumerate(hidden_sizes):
        autoencoder = Autoencoder(input_size, hidden_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

        print(f'Начало обучения автоэнкодера с скрытым слоем размера {hidden_size} (Слой {idx + 1}/{len(hidden_sizes)})')
        print(f'Входные нейроны: {input_size}, Выходные нейроны: {hidden_size}')

        for epoch in range(10):
            autoencoder.train()
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()

            print(f'Слой {idx + 1}, Эпоха: {epoch + 1}, Loss: {loss.item():.4f}')

        pretrained_weights.append(autoencoder.encoder)

        autoencoder.eval()
        with torch.no_grad():
            _, data = autoencoder(data)

        input_size = hidden_size

    return pretrained_weights


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=["RiskLevel"]).values
    y = data["RiskLevel"].factorize()[0]
    return X, y


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_mlp(model, X_train_tensor, y_train_tensor, epochs=500):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_values = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'MLP Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        loss_values.append(loss.item())
    
    return loss_values


def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        _, y_pred = torch.max(y_pred_logits, 1)

    accuracy = (y_pred == y_test_tensor).float().mean()
    return accuracy.item()


def plot_results(y_test_tensor, y_pred, num_classes):
    plt.figure(figsize=(10, 6))
    plt.hist(y_test_tensor.numpy(), bins=num_classes, alpha=0.5, label='True')
    plt.hist(y_pred.numpy(), bins=num_classes, alpha=0.5, label='Predicted')
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
    X, y = load_and_preprocess_data(r'IAD/lab3/Maternal Health Risk Data Set.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))

    hidden_sizes = [64, 32, 16, 8]
    pretrained_weights = pretrain_autoencoders(X_train_tensor, input_size, hidden_sizes)

    mlp_model = MLP(input_size, num_classes, pretrained_weights)
    loss_values = train_mlp(mlp_model, X_train_tensor, y_train_tensor)

    accuracy = evaluate_model(mlp_model, X_test_tensor, y_test_tensor)
    print(f'Accuracy: {accuracy:.2f}')

    plot_results(y_test_tensor, mlp_model(X_test_tensor).argmax(dim=1), num_classes)
    plot_loss(loss_values, 500)


if __name__ == "__main__":
    main()
