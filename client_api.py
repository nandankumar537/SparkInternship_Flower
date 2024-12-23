from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import flwr as fl  # Updated Flower import
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Create FastAPI app
app = FastAPI()

# Define FlowerClient class
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, data, epochs, batch_size, verbose):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

# Define REST API models
class ClientConfig(BaseModel):
    partition_id: int
    num_partitions: int
    local_epochs: int
    batch_size: int
    verbose: bool = False

# Dummy data and model loading function (replace with actual logic)
def load_data(partition_id, num_partitions):
    # Dummy data, replace with actual partitioning logic
    x_train = np.random.rand(100, 32)
    y_train = np.random.randint(0, 2, 100)
    x_test = np.random.rand(50, 32)
    y_test = np.random.randint(0, 2, 50)
    return (x_train, y_train, x_test, y_test)

def load_model():
    # Dummy model, replace with actual model loading
    model = Sequential([
        Dense(64, activation='relu', input_shape=(32,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define client_fn for Flower
def client_fn(context: fl.server.ClientContext):
    model = load_model()
    data = load_data(0, 1)  # Replace with partition logic
    epochs = 1  # Adjust as needed
    batch_size = 32  # Adjust as needed
    verbose = 1  # Adjust as needed
    return FlowerClient(model, data, epochs, batch_size, verbose)

# Define API endpoints
@app.post("/start_client")
def start_client(config: ClientConfig):
    try:
        # Simulate context creation for the Flower client
        context = fl.server.ClientContext(
            node_config={
                "partition-id": config.partition_id,
                "num-partitions": config.num_partitions,
            },
            run_config={
                "local-epochs": config.local_epochs,
                "batch-size": config.batch_size,
                "verbose": config.verbose,
            },
        )
        # Initialize Flower ClientApp
        client_app = fl.client.ClientApp(client_fn=client_fn)
        return {"status": "Client initialized and ready to connect"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fit")
def fit(parameters: Dict[str, Any]):
    try:
        # Load model and data for training
        model = load_model()
        data = load_data(0, 1)  # Replace with partition logic
        client = FlowerClient(model, data, epochs=1, batch_size=32, verbose=1)
        new_weights, num_samples, metrics = client.fit(parameters, config={})
        return {"new_weights": new_weights, "num_samples": num_samples, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
def evaluate(parameters: Dict[str, Any]):
    try:
        # Load model and data for evaluation
        model = load_model()
        data = load_data(0, 1)  # Replace with partition logic
        client = FlowerClient(model, data, epochs=1, batch_size=32, verbose=1)
        loss, num_samples, metrics = client.evaluate(parameters, config={})
        return {"loss": loss, "num_samples": num_samples, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
