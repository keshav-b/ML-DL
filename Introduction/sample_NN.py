import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		self.hidden = nn.Linear(2,2)
		self.output = nn.Linear(1,1)

	def forward(self, x):
		x = self.hidden(x)
		x = torch.tanh(x)
		x = self.output(x)
		x = torch.softmax(x)
		return x

model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for i in range(epochs):
	y_pred = model(x)
	loss = criterion(y_pred, y)
	'''
	loss.item() gives the loss
	total_loss +=loss.item()
	'''
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# loss_history.append(total_loss/no_of_samples)	
