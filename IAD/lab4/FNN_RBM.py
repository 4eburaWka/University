import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=["motor_UPDRS"])
    y = data["motor_UPDRS"].values
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))
        self.v_bias = nn.Parameter(torch.zeros(visible_units))

    def sample_h(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return torch.bernoulli(h_prob)

    def sample_v(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return torch.bernoulli(v_prob)

    def contrastive_divergence(self, v, k=1):
        v0 = v
        for _ in range(k):
            h_sample = self.sample_h(v)
            v = self.sample_v(h_sample)
        return v0, v

    def train_rbm(self, data, lr=0.01, batch_size=64, epochs=10):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                v0, vk = self.contrastive_divergence(batch)
                loss = torch.mean((v0 - vk) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        print("RBM training complete.")


class MLP(nn.Module):
    def __init__(self, input_size, rbm_weights):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc1.weight.data = rbm_weights  # Инициализация весов из RBM
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Функция для обучения MLP
def train_model(
    model, criterion, optimizer, X_train_tensor, y_train_tensor, num_epochs=500
):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        mape = mean_absolute_percentage_error(y_test_tensor.numpy(), y_pred.numpy())
        return y_pred, mape


def plot_results(y_test, y_pred):
    errors = np.abs(y_test - y_pred).flatten()

    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(y_test, y_pred, c=errors, cmap="coolwarm", alpha=0.6)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        lw=2,
        linestyle="-",
    )
    plt.colorbar(scatter, label="Ошибка предсказания")
    plt.xlabel("Истинные значения")
    plt.ylabel("Предсказанные значения")
    plt.title("Сравнение предсказанных значений с истинными")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X, y = load_data(r"IAD\lab4\parkinsons_updrs.data")
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

    rbm = RBM(visible_units=X_train.shape[1], hidden_units=64)
    rbm.train_rbm(X_train_tensor, lr=0.03, batch_size=64, epochs=500)

    model = MLP(input_size=X_train.shape[1], rbm_weights=rbm.W)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_model(
        model, criterion, optimizer, X_train_tensor, y_train_tensor, num_epochs=500
    )

    y_pred, mape = evaluate_model(model, X_test_tensor, y_test_tensor)
    print(f"MAPE: {mape:.4f}")

    plot_results(y_test_tensor.numpy(), y_pred.numpy())
