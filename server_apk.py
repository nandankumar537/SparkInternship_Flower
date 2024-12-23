"""flwr: A Flower / TensorFlow REST API server."""

from fastapi import FastAPI, HTTPException
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.task import load_model

# Initialize FastAPI app
app = FastAPI()

# Define global variables
server_app = None
is_server_running = False


def server_fn(context: Context):
    """Flower server function to configure strategy and server."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


@app.on_event("startup")
def startup_event():
    """Initialize the Flower server on startup."""
    global server_app
    global is_server_running
    if not is_server_running:
        server_app = ServerApp(server_fn=server_fn)
        is_server_running = True


@app.get("/")
def read_root():
    """Root endpoint for server."""
    return {"message": "Welcome to the Flower Federated Learning REST API Server"}


@app.get("/server/status")
def get_server_status():
    """Check if the server is running."""
    return {"server_running": is_server_running}


@app.post("/server/start")
def start_server():
    """Start the Flower server."""
    global server_app
    global is_server_running

    if is_server_running:
        return {"message": "Server is already running"}

    try:
        server_app.start()
        is_server_running = True
        return {"message": "Server started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")


@app.post("/server/stop")
def stop_server():
    """Stop the Flower server."""
    global server_app
    global is_server_running

    if not is_server_running:
        return {"message": "Server is not running"}

    try:
        server_app.stop()
        is_server_running = False
        return {"message": "Server stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop server: {str(e)}")

