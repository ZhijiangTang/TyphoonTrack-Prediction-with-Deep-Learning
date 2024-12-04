import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data import MyData
from model import MyModel2
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


def train():
	batch_size = 20000
	
	setup_seed(10086)
	traindata = MyData(data_path='./data/TrainData.json', l=4, frac=1)
	traindata.get_mix_t_sequence()
	# traindata.get_t_sequence(t=1)
	valdata = MyData(data_path='./data/TestData.json', l=4)
	valdata.get_mix_t_sequence()
	# valdata.get_t_sequence(t=1)
	train_dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(valdata, batch_size=batch_size, shuffle=False)
	
	model = MyModel2(name=f'Model2').to(device)
	criterion = nn.MSELoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	# 定义余弦退火调度器
	num_epochs = 20
	scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
	
	# 进行模型训练
	epochs = 5000
	best_rmse = 1
	for epoch in range(epochs):
		model.train()
		print(f'==========Epoch {epoch}==========')
		for idx, batch in enumerate(train_dataloader):
			X, y, _, info = batch
			# print(info[-1].shape)
			t = torch.unsqueeze(info[-1], dim=1).to(torch.float).to(device)
			
			X = X.to(device)
			y = y.to(device)
			
			# 前向传播和计算损失
			optimizer.zero_grad()
			outputs = model(X, t)
			loss = criterion(outputs, y)
			# print(f'{idx}, Loss: {loss.item()}')
			
			# 反向传播和参数更新
			loss.backward()
			optimizer.step()
			scheduler.step()
		
		for idx, batch in enumerate(val_dataloader):
			X, y, _, _ = batch
			t = torch.tensor([[1.0]]).to(device)
			X = X.to(device)
			y = y.to(device)
			outputs = model(X, t)
			# loss = criterion(outputs, y)
			RMSE = torch.sqrt(torch.mean(torch.sum(torch.square(outputs[:, :2] - y[:, :2]), dim=1)))
			
			if best_rmse > RMSE:
				best_rmse = RMSE
				print('Test Best RMSE', RMSE.item())
				model.save(f'best.pth')


def test(model_name, t):
	testdata = MyData(data_path='./data/TestData.json', l=4, frac=1)
	testdata.get_t_sequence(t=t)
	batch_size = 10000
	RMSE = {}
	
	GT_Predict = pd.DataFrame(np.full([len(testdata), 5], np.nan),
	                          columns=['True_lat', 'True_lon', 'Predict_lat', 'Predict_lon', 'SE'])
	
	model = MyModel2().to(device)
	model.load_state_dict(torch.load(f'checkpoints/{model_name}/best.pth'))
	test_dataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)
	
	for i, batch in enumerate(test_dataloader):
		X, y, _, info = batch
		t = torch.unsqueeze(info[-1], dim=1).to(torch.float).to(device)
		
		predict_y = testdata.inverse_norm(model(X.to(device), t).cpu().detach()).numpy()
		true_y = testdata.inverse_norm(y).numpy()
	GT_Predict.iloc[:, :2] = true_y[:, :2] * 0.1
	
	GT_Predict.iloc[:, 2:4] = predict_y[:, :2] * 0.1
	GT_Predict.iloc[:, 4] = np.sum(np.square(GT_Predict.iloc[:, :2].values - GT_Predict.iloc[:, 2:4].values), axis=1)
	
	GT_Predict.to_csv(f'checkpoints/Model/TestPredict.csv')
	RMSE[f'Model'] = np.sqrt(np.mean(GT_Predict['SE'].values))
	print(RMSE)


def test_19(model_name):
	testdata = MyData(data_path='./data/TestData.json', l=4, frac=1, TaifengID=19)
	batch_size = 4096
	RMSE = {}
	
	GT_Predict = pd.DataFrame(np.full([len(testdata), 5], np.nan),
	                          columns=['True_lat', 'True_lon', 'Predict_lat', 'Predict_lon', 'SE'])
	
	model = MyModel2().to(device)
	model.load_state_dict(torch.load(f'checkpoints/{model_name}/best.pth'))
	test_dataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)
	
	for i, batch in enumerate(test_dataloader):
		X, y, last, _ = batch
		
		t = testdata.encoded_sequence[0]
		t = t.repeat(len(testdata), 1, 1).to(device)
		
		model.eval()
		predict_y = testdata.inverse_norm(model(X.to(device), t).cpu().detach()).numpy()
		true_y = testdata.inverse_norm(y).numpy()
	GT_Predict.iloc[:, :2] = true_y[:, :2] * 0.1
	
	GT_Predict.iloc[:, 2:4] = predict_y[:, :2] * 0.1
	GT_Predict.iloc[:, 4] = np.sum(np.square(GT_Predict.iloc[:, :2].values - GT_Predict.iloc[:, 2:4].values), axis=1)
	
	# GT_Predict.to_csv(f'checkpoints/24h/TestPredict.csv')
	# RMSE[f'24h-2'] = np.sqrt(np.mean(GT_Predict['SE'].values))
	# RMSE
	GT_Predict.to_csv(f'checkpoints/{model_name}/{model_name}_19Taifeng.csv')
	print(GT_Predict)
	
	# fig = plt.figure(figsize=(8, 4))
	# ax = fig.add_subplot()
	# plt.scatter(GT_Predict['True_lon'], GT_Predict['True_lat'], color=mycolors[1])
	# plt.scatter(GT_Predict['Predict_lon'], GT_Predict['Predict_lat'], color=mycolors[3])
	# # plt.scatter(y_predict[:,1],y_predict[:,0],color=mycolors[9])
	# plt.legend(['True', 'Predict'])
	# plt.show()


if __name__ == '__main__':
	model_name = 'Model2'
	# train()
	# test(model_name,t=1)
	# test(model_name, t=4)
	# test(model_name, t=8)
	# test(model_name, t=12)
	test_19(model_name)

	print('end')
