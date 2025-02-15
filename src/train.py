from src.model import build_model
from src.data_preprocessing import load_data

X_train, y_train = load_data("train_data/")
model = build_model(input_shape=(128, 128, 1), num_classes=10)  # Ejemplo para dígitos
model.fit(X_train, y_train, epochs=10, validation_split=0.2)