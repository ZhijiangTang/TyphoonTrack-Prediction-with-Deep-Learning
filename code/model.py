import math

import torch
import torch.nn as nn
from data import MyData
import os
from torch.nn.utils.rnn import pad_packed_sequence


class BPNetModel(torch.nn.Module):
	def __init__(self, n_feature=16, n_hidden=30, n_output=4, name='BaseMLP'):
		super(BPNetModel, self).__init__()
		self.name = name
		self.hiddden = torch.nn.Linear(n_feature, n_hidden)  # 定义隐层网络
		self.out = torch.nn.Linear(n_hidden, n_output)  # 定义输出层网络
	
	def forward(self, x):
		batch_size, seq_len, feature_len = x.shape
		x = torch.reshape(x, (batch_size, -1))
		x = torch.tanh(self.hiddden(x))  # 隐层激活函数采用relu()函数
		out = self.out(x)
		return out
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


class MyLSTMDiSeqlen(nn.Module):
	def __init__(self, input_size=4, hidden_size=256, output_size=4, name='BaseLSTM'):
		super(MyLSTMDiSeqlen, self).__init__()
		self.hidden_size = hidden_size
		self.name = name
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
		self.mlp = nn.Sequential(nn.Linear(hidden_size * 2, 1024),
		                         nn.ReLU(),
		                         nn.Dropout(0.1),
		                         nn.Linear(1024, output_size))
	
	def forward(self, seq, last):
		out, _ = self.lstm(seq)
		out = out[:, -1, :]
		
		output = self.mlp(out)
		return output
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


class BaseLSTM(nn.Module):
	def __init__(self, input_size=4, hidden_size=256, output_size=4, name='BaseLSTM'):
		super(BaseLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.name = name
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
		self.mlp = nn.Sequential(nn.Linear(hidden_size, 1024),
		                         nn.ReLU(),
		                         nn.Linear(1024, output_size))
	
	def forward(self, seq):
		out, _ = self.lstm(seq)
		out = out[:, -1, :]
		
		output = self.mlp(out)
		return output
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


class MyBidirLSTM(nn.Module):
	def __init__(self, input_size=4, hidden_size=256, output_size=4, name='BaseBidirLSTM'):
		super(MyBidirLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.name = name
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
		self.mlp = nn.Sequential(nn.Linear(hidden_size * 2, 1024),
		                         nn.ReLU(),
		                         nn.Linear(1024, output_size))
	
	def forward(self, seq):
		out, _ = self.lstm(seq)
		out = out[:, -1, :]
		
		output = self.mlp(out)
		return output
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


class MyL2Layer(nn.Module):
	def __init__(self, input_size=4, hidden_size=256, output_size=4, name='BaseLSTM'):
		super(MyL2Layer, self).__init__()
		self.hidden_size = hidden_size
		self.name = name
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=False, batch_first=True)
		self.mlp = nn.Sequential(nn.Linear(hidden_size, 1024),
		                         nn.ReLU(),
		                         # nn.Dropout(0.1),
		                         nn.Linear(1024, output_size))
	
	def forward(self, seq):
		out, _ = self.lstm(seq)
		out = out[:, -1, :]
		
		output = self.mlp(out)
		return output
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))

class MyLSTM(nn.Module):
	def __init__(self, input_size=4, hidden_size=128, output_size=4, name='BaseLSTM'):
		super(MyLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.name = name
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
		self.mlp = nn.Sequential(nn.Linear(hidden_size * 2, 1024),
		                         nn.ReLU(),
		                         # nn.Dropout(0.1),
		                         nn.Linear(1024, output_size))
	
	def forward(self, seq):
		out, _ = self.lstm(seq)
		out = out[:, -1, :]
		
		output = self.mlp(out)
		return output
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


class MyChallenge(nn.Module):
	def __init__(self, input_size=4, hidden_size=124, output_size=4, name='BaseChallenge'):
		super(MyChallenge, self).__init__()
		self.hidden_size = hidden_size
		self.name = name
		self.Decoder = nn.LSTM(input_size, hidden_size, batch_first=True,bidirectional=False)
		self.Encoder = nn.LSTM(hidden_size+4, 32, batch_first=True)
		self.mlp = nn.Sequential(nn.Linear(32, 1024),
		                         nn.ReLU(),
		                         nn.Linear(1024, output_size))
	
	def forward(self, seq):
		out, _ = self.Decoder(seq)
		out = torch.cat([seq, out], dim=2)
		out, _ = self.Encoder(out)
		
		out = out[:, -1, :]
		out = self.mlp(out)
		return out
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_seq_len):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=0.1)
		
		pe = torch.zeros(max_seq_len, d_model)
		position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		# print(self.pe[:x.size(1),-1, :].shape)
		x = x + self.pe[:x.size(1), -1, :]
		return self.dropout(x)


class MyTransformer(nn.Module):
	def __init__(self, input_size=4, output_size=4, num_layers=2, num_heads=4, hidden_size=128, name='BaseTransformer'):
		super(MyTransformer, self).__init__()
		self.name = name
		# self.position_encoding = PositionalEncoding(input_size, 4)
		self.transformer_encoder = nn.TransformerEncoder(
			nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, batch_first=True),
			num_layers
		)
		self.mlp = nn.Sequential(nn.Linear(input_size, 1024),
		                         nn.ReLU(),
		                         nn.Linear(1024, output_size))
	
	def forward(self, seq):
		# seq = self.position_encoding(seq)  # 添加位置编码
		out = self.transformer_encoder(seq)
		out = self.mlp(out[:, -1, :])  # 取最后一个时间步的输出进行预测
		return out
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


class Myattention(nn.Module):
	def __init__(self, input_size=4, hidden_size=64, output_size=4, name='BaseModel'):
		super(Myattention, self).__init__()
		self.hidden_size = hidden_size
		self.attention = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
		self.QLinear = nn.Linear(input_size, hidden_size)
		self.KLinear = nn.Linear(input_size, hidden_size)
		self.VLinear = nn.Linear(input_size, hidden_size)
	
	def forward(self, Q, K, V):
		return self.attention(self.QLinear(Q), self.KLinear(K), self.VLinear(V))


class T_embeding(nn.Module):
	def __init__(self, input_size=1, hidden_size1=4, hidden_size2=256, hidden_size3=128, output_size=4,
	             name='BaseModel'):
		super(T_embeding, self).__init__()
		self.relu = nn.ReLU()
		self.linear1 = nn.Linear(input_size, hidden_size1)
		self.linear2 = nn.Linear(hidden_size1, hidden_size3)
	
	def forward(self, t):
		# print(t)
		out1 = self.linear1(t)
		out2 = self.linear2(self.relu(out1))
		return torch.unsqueeze(out2, dim=1)


class MyModel(nn.Module):
	def __init__(self, input_size=4, hidden_size=128, output_size=4, name='BaseModel'):
		super(MyModel, self).__init__()
		self.hidden_size = hidden_size
		self.name = name
		
		self.t_embeding = T_embeding(input_size=hidden_size, hidden_size1=1024, hidden_size2=hidden_size,
		                             hidden_size3=hidden_size)
		self.Tattention = Myattention(input_size=input_size, hidden_size=64)
		self.Decoder = nn.LSTM(64, hidden_size, batch_first=True)
		self.Seqattention = Myattention(input_size=hidden_size, hidden_size=64)
		self.Encoder = nn.LSTM(64, hidden_size, batch_first=True)
		
		self.mlp = nn.Sequential(nn.Linear(hidden_size, 1024),
		                         nn.ReLU(),
		                         nn.Linear(1024, output_size))
	
	def forward(self, seq, t):
		
		# print(t_embeding1.shape, seq.shape)
		out, _ = self.Tattention(seq, seq, seq)
		out_d, _ = self.Decoder(out)
		t_out = out_d + t
		t_out = t_out[:, -1, :]
		t_out = torch.unsqueeze(t_out, dim=1)
		t_embeding = self.t_embeding(t_out)
		
		out_sa, _ = self.Seqattention(out_d, out_d, out_d)
		out_e, _ = self.Encoder(out_sa)
		# print(out_e)
		
		out = out_e + t_embeding
		out += t_out
		# print(t_embeding3.shape,out.shape)
		out = out[:, -1, :]
		out = self.mlp(out)
		return out
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


class MyModel2(nn.Module):
	def __init__(self, input_size=4, hidden_size=128, output_size=4, name='BaseModel'):
		super(MyModel2, self).__init__()
		self.hidden_size = hidden_size
		self.name = name
		
		# self.t_embeding = T_embeding(input_size=hidden_size, hidden_size1=1024, hidden_size2=hidden_size,
		#                              hidden_size3=hidden_size)
		# self.Tattention = Myattention(input_size=input_size, hidden_size=64)
		self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
		# self.Seqattention = Myattention(input_size=hidden_size, hidden_size=64)
		# self.lstm = nn.LSTM(64, 32, batch_first=True)
		
		self.mlp = nn.Sequential(nn.Linear(hidden_size, 1024),
		                         nn.ReLU(),
		                         nn.Linear(1024, output_size))
	
	def forward(self, seq, t):
		# out, _ = self.Tattention(seq, seq, seq)
		out, _ = self.lstm(seq)
		# t_out = out_d+t
		# t_embeding = self.t_embeding(t_out[:,-1,:])
		
		out = out + t
		out = out[:, -1, :]
		out = self.mlp(out)
		return out
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


class MyDecoderEncoder(nn.Module):
	def __init__(self, input_size=4, hidden_size=128, output_size=4, name='BaseModel'):
		super(MyDecoderEncoder, self).__init__()
		self.hidden_size = hidden_size
		self.name = name
		self.Decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
		
		self.attention = nn.MultiheadAttention(32, num_heads=1, batch_first=True)
		self.QLinear = nn.Linear(hidden_size, 32)
		self.KLinear = nn.Linear(hidden_size, 32)
		self.VLinear = nn.Linear(hidden_size, 32)
		
		self.Encoder = nn.LSTM(32, 128, batch_first=True)
		
		self.mlp = nn.Sequential(nn.Linear(128, 512),
		                         nn.ReLU(),
		                         nn.Linear(512, output_size))
	
	def forward(self, seq):
		out, _ = self.Decoder(seq)
		
		out, attention_weights = self.attention(self.QLinear(out), self.KLinear(out), self.VLinear(out))
		out, _ = self.Encoder(out)
		out = out[:, -1, :]
		output = self.mlp(out)
		return output
	
	def save(self, model_name=None):
		if model_name is None:
			model_name = self.name + '.pth'
		save_path = os.path.join('checkpoints', f'{self.name}')
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		torch.save(self.state_dict(), os.path.join(save_path, f'{model_name}'))


if __name__ == '__main__':
	data = MyData(data_path='./data/TrainData.json')
	X, y, last, info = data[24524]
	X = torch.unsqueeze(X, dim=0)
	model = MyLSTM()
	predict = model(X, last)
	print(predict)
	print('end')
