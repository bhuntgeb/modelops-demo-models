import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
import onnx
import torch
from torch import nn

# define a torch model
net = nn.Linear(6, 1)
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

X = torch.randn(6)
y = torch.randn(1)

# run model training
epochs = 5
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(X)

    loss = loss_function(outputs, y)
    loss.backward()

    optimizer.step()

# convert model to ONNX and load it
torch.onnx.export(net, X, "model.onnx")
onnx_model = onnx.load_model("model.onnx")

# log the model into a mlflow run
with mlflow.start_run():
    signature = infer_signature(X.numpy(), net(X).detach().numpy())
    model_info = mlflow.onnx.log_model(onnx_model, "model", signature=signature)

# load the logged model and make a prediction
onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = onnx_pyfunc.predict(X.numpy())
print(predictions)
