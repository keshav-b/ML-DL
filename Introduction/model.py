import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
	def __init__(self, inputs=3, hidden1=2, outputs=1):
		super().__init__()
		self.fc1 = nn.Linear(inputs, hidden1)
		self.out = nn.Linear(hidden1, outputs)

	def forward(self, x):
		x = self.fc1(x)
		x = F.tanh(x)
		x = self.out(x)
		x = self.softmax(x, dim=0)
		return x

model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# define x_train, y_train, x_test, y_test 
losses = []

for i in range(epochs):
	y_pred = model(x_train)  # or model.forward(x_train)

	loss = criterion(y_pred, y_train)
	losses.append(loss)
	'''
	if(i%10 == 0):
		print(f'Epoch: {i} | Loss: {loss}')
	'''
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


