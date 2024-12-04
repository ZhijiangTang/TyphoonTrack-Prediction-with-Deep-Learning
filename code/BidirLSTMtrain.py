import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data import MyData, collate_fn
from model import MyBidirLSTM
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

mycolors = plt.cm.RdBu(list(range(0, 256, 256 // 10)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model_name):
	batch_size = 15000
	
	setup_seed(10086)
	traindata = MyData(data_path='./data/TrainData.json', l=4, frac=1)
	valdata = MyData(data_path='./data/TestData.json', l=4)
	
	train_dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(valdata, batch_size=batch_size, shuffle=False)
	
	model = MyBidirLSTM(name=f'{model_name}').to(device)
	criterion = nn.MSELoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	# 定义余弦退火调度器
	num_epochs = 20
	scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
	
	# 进行模型训练
	epochs = 20000
	history = pd.DataFrame(np.full([epochs, 3], np.nan), columns=['Epoch', 'Loss', 'Val RMSE'])
	best_rmse = 1
	for epoch in range(epochs):
		model.train()
		print(f'==========Epoch {epoch}==========')
		loss_train = 0
		for idx, batch in enumerate(train_dataloader):
			X, y, _, _ = batch
			X = X.to(device)
			y = y.to(device)
			
			# 前向传播和计算损失
			optimizer.zero_grad()
			outputs = model(X)
			loss = criterion(outputs, y)
			# print(f'{idx}, Loss: {loss.item()}')
			
			# 反向传播和参数更新
			loss.backward()
			loss_train += loss.item()
			optimizer.step()
			scheduler.step()
		loss_train /= (idx+1)
		
		for idx, batch in enumerate(val_dataloader):
			X, y, _, _ = batch
			X = X.to(device)
			y = y.to(device)
			outputs = model(X)
			loss = criterion(outputs, y)
			RMSE = torch.sqrt(torch.mean(torch.sum(torch.square(outputs[:, :2] - y[:, :2]), dim=1)))
			
			if best_rmse > RMSE:
				best_rmse = RMSE
				print('Test Best RMSE', RMSE.item())
				model.save(f'best.pth')
		history.iloc[epoch,:] = [epoch, loss_train, RMSE.item()]
	history.to_csv(f'checkpoints/{model.name}/history.csv')


def test(model_name):
	testdata = MyData(data_path='./data/TestData.json', l=4, frac=1)
	batch_size = 4096
	RMSE = {}
	
	GT_Predict = pd.DataFrame(np.full([len(testdata), 5], np.nan),
	                          columns=['True_lat', 'True_lon', 'Predict_lat', 'Predict_lon', 'SE'])
	
	model = MyBidirLSTM().to(device)
	model.load_state_dict(torch.load(f'checkpoints/{model_name}/best.pth'))
	test_dataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)
	
	for i, batch in enumerate(test_dataloader):
		X, y, _, _ = batch
		
		predict_y = testdata.inverse_norm(model(X.to(device)).cpu().detach()).numpy()
		true_y = testdata.inverse_norm(y).numpy()
	GT_Predict.iloc[:, :2] = true_y[:, :2] * 0.1
	
	GT_Predict.iloc[:, 2:4] = predict_y[:, :2] * 0.1
	GT_Predict.iloc[:, 4] = np.sum(np.square(GT_Predict.iloc[:, :2].values - GT_Predict.iloc[:, 2:4].values), axis=1)
	
	GT_Predict.to_csv(f'checkpoints/{model_name}/TestPredict.csv')
	RMSE[f'24h'] = np.sqrt(np.mean(GT_Predict['SE'].values))
	print(RMSE)


def test_19(model_name):
	testdata = MyData(data_path='./data/TestData.json', l=4, frac=1, TaifengID=19)
	batch_size = 4096
	RMSE = {}
	
	GT_Predict = pd.DataFrame(np.full([len(testdata), 5], np.nan),
	                          columns=['True_lat', 'True_lon', 'Predict_lat', 'Predict_lon', 'SE'])
	
	model = MyBidirLSTM().to(device)
	model.load_state_dict(torch.load(f'checkpoints/{model_name}/best.pth'))
	test_dataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)
	
	for i, batch in enumerate(test_dataloader):
		X, y, _, _ = batch
		model.eval()
		predict_y = testdata.inverse_norm(model(X.to(device)).cpu().detach()).numpy()
		true_y = testdata.inverse_norm(y).numpy()
	GT_Predict.iloc[:, :2] = true_y[:, :2] * 0.1
	
	GT_Predict.iloc[:, 2:4] = predict_y[:, :2] * 0.1
	GT_Predict.iloc[:, 4] = np.sum(np.square(GT_Predict.iloc[:, :2].values - GT_Predict.iloc[:, 2:4].values), axis=1)
	
	# GT_Predict.to_csv(f'checkpoints/24h/TestPredict.csv')
	RMSE[f'24h-2'] = np.sqrt(np.mean(GT_Predict['SE'].values))
	# RMSE
	print(GT_Predict)
	
	fig = plt.figure(figsize=(8, 4))
	ax = fig.add_subplot()
	plt.scatter(GT_Predict['True_lon'], GT_Predict['True_lat'], color=mycolors[1])
	plt.scatter(GT_Predict['Predict_lon'], GT_Predict['Predict_lat'], color=mycolors[3])
	# plt.scatter(y_predict[:,1],y_predict[:,0],color=mycolors[9])
	plt.legend(['True', 'Predict'])
	plt.show()


if __name__ == '__main__':
	model_name = 'BidirLSTM'
	train(model_name)
	test(model_name)
	test_19(model_name)
	print('end')
