# IMPORT LIBRARIES
import torch
import torch.nn as nn
import torch.nn.functional as F

# MODULE CREATION
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

# INSTANTIATE PARAMETERS
model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# TRAINING
# define x_train, y_train, x_test, y_test 
losses = []
for i in range(epochs):
	y_pred = model(x_train)  # or model.forward(x_train)

	loss = criterion(y_pred, y_train)
	losses.append(loss)
	
	if(i%5 == 0):
		print(f'Epoch: {i} | Loss: {loss}')
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

# VALIDATION	
with torch.no_grad(): # deactivate the auto gradient engine, no back prop
	y_eval = model.forward(x_test)
	loss = criterion(y_eval, y_test)
# print(loss)

# VIEWING THE RESULTS OF THE TEST DATA
with torch.no_grad():

	print("Model Predicted \t\t\t Actual Output")
	for data in x_test:
		y_val = model.forward(data)

		print(f"{str(y_val)} {y_test[i]}")


# SAVING THE MODEL
torch.save(model.state_dict(), "sample_model.pt")		

# USING THE SAVED MODEL
my_model = Network()
my_model.load_state_dict(torch.load("sample_model.pt"))

# TESTING WITH NEWLY LOADED MODEL
new_test = torch.tensor('''[new test data]''' )
with torch.no_grad():
	print(my_model(new_test))



		

	



