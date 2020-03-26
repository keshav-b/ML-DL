import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
	def __init__(self, inputs=4, hidden1=8, hidden2=9, outputs=3):
		super().__init__()
		self.fc1 = nn.Linear(inputs, hidden1)
		self.fc2 = nn.Linear(hidden1, hidden2)
		self.out = nn.Linear(hidden2, outputs)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.out(x)
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


