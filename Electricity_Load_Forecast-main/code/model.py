import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import random
import func 
import torch.nn.functional as F
import copy
import math
import xgboost as xgb
import argparse

# 固定随机种子，保持每次模型训练结果的精度一致
os.environ['KMP_DUPLICATE_LIB_OK']='True'
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="运行模型脚本")
parser.add_argument('--config_path', type=str, required=True, help='配置文件路径')
args = parser.parse_args()

# 使用传递的配置文件路径
config_path = args.config_path
# config_path = 'config_96.json'
CONFIG = func.load_config(config_path)
func.print_config(config_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = StandardScaler()

data_path = CONFIG['COMMON']['data_path']  
seq_length = CONFIG['COMMON']['seq_length']
input_size = CONFIG['COMMON']['input_size']
pred_length = CONFIG['COMMON']['pred_length']
output_size = CONFIG['COMMON']['pred_length']
model_name = CONFIG['COMMON']['model_name']
save_data_path = os.path.join(CONFIG['COMMON']['save_path'], model_name)
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

# 数据处理
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.load_data = data[:,0,0].reshape(-1, 1)
        self.weather_time_data = data[:,0,1:]
        self.seq_length = seq_length
        self.pred_length = pred_length 

    def __len__(self):
        return len(self.load_data) - self.seq_length - self.pred_length

    def __getitem__(self, index):
        # x为负荷数据和天气时间数据的拼接，y为负荷数据
        x_load = self.load_data[index:index + self.seq_length]
        x_weather_time = self.weather_time_data[index:index + self.seq_length]
        x_combined = np.concatenate([x_load, x_weather_time], axis=1)
        y = self.load_data[index + self.seq_length:index + self.seq_length + self.pred_length]
        y = y.reshape(-1)
        return x_combined, y
    
# 数据处理   
def get_dataloaders(model_name):# 读取训练集和验证集的数据
    batch_size = CONFIG[model_name]['batch_size']

    df = pd.read_csv(data_path)
    m_all = len(df) # 数据集总行数
    m_val = 52*96 # 验证集数量
    m_test = 52*96 # 测试集数量
    m_train = m_all - m_test - m_val # 训练集数量
    # print(m_train)

    # Avg_Temperature  Avg_Humidity Rainfall缺失值用前一个值填充
    df['Avg_Temperature'] = df['Avg_Temperature'].fillna(method='ffill')
    df['Avg_Humidity'] = df['Avg_Humidity'].fillna(method='ffill')
    df['Rainfall'] = df['Rainfall'].fillna(method='ffill')
    df = df[df['Time'] <= '2015-01-11']
    df = df.drop('Time', axis=1)

    train_df = df.iloc[:m_train]
    val_df = df.iloc[m_train:m_train+m_val]
    test_df = df.iloc[m_train+m_val:]

    
    # 用训练集的数据计算均值和方差
    test_df_origin = test_df

    if model_name != "XGBoost":
        scaler.fit(train_df)
        train_df = scaler.transform(train_df)
        # 对验证集和测试集进行相同的归一化操作
        val_df = scaler.transform(val_df)
        test_df = scaler.transform(test_df)
    else:
        train_df = train_df.values
        val_df = val_df.values
        test_df = test_df.values

    print('Dimensions of training, validation, and test sets:')
    print(train_df.shape, val_df.shape, test_df.shape)

    # 将数据集转换为PyTorch的Tensor
    train_data = torch.tensor(train_df, dtype=torch.float32).unsqueeze(1)
    val_data = torch.tensor(val_df, dtype=torch.float32).unsqueeze(1)
    test_data = torch.tensor(test_df, dtype=torch.float32).unsqueeze(1)

    train_dataset = TimeSeriesDataset(train_data, seq_length,pred_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length,pred_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length, pred_length)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_df_origin, train_dataloader, val_dataloader, test_dataloader

# 根据傅里叶变换算出时间序列的周期
def FFT_for_Period(x, k=2):
    # [B, T, C]
    x = x[:,:,0].unsqueeze(2)
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

# 1D-CNN网络定义 
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length, kernel_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.kernel_size = kernel_size

        # 创建多个卷积层
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(Conv1dLayer(input_size, kernel_size,hidden_size))
            else:
                layers.append(Conv1dLayer(hidden_size, kernel_size,hidden_size))
        self.convLayers = nn.ModuleList(layers)

        # 定义全连接层
        # self.fc = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(self.seq_length//(2**num_layers), output_size)

    def forward(self, x):
        B, T, N = x.size()
        # x = x.transpose(1, 2)  # 调整维度以符合卷积层的输入要求 (B, N, T)

        # 应用多个卷积层
        for layer in self.convLayers:
            x = layer(x)

        x = x.mean(dim=2)  # 全局平均池化, (B, hidden_size)
        x = self.fc(x)  # 应用全连接层
        return x
    
# GRU网络定义
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, N = x.size()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    

# LSTM网络定义
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, N = x.size()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# CNN-LSTM网络定义
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length,  kernel_size):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.kernel_size = kernel_size
        self.conv1dLayer = Conv1dLayer(input_size, kernel_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.attention = Attention(hidden_size)
        

    def forward(self, x):
        B, T, N = x.size()
        x = self.conv1dLayer(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class CNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length, kernel_size):
        super(CNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.kernel_size = kernel_size
        self.conv1dLayer = Conv1dLayer(input_size,kernel_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, N = x.size()
        x = self.conv1dLayer(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 一维卷积层定义   
class Conv1dLayer(nn.Module):
    def __init__(self, c_in,kernel_size=3,c_out=None):
        super(Conv1dLayer, self).__init__()
        if c_out is None:
            c_out = c_in
        padding = (kernel_size-1)//2 
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_out,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_out)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x
    
# 2D-CNN-LSTM定义 
class TwoD_CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size):
        super(TwoD_CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.main_period = main_period
        self.kernel_size = kernel_size
        
        self.conv2d = nn.Conv2d(input_size, input_size, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.conv1dLayer = Conv1dLayer(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)      

    def forward(self, x):
        B, T, N = x.size()
        # period,_ = FFT_for_Period(x,1)
        # print(period)
        x = x.reshape(B,  self.seq_length//self.main_period ,self.main_period, N).permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, N)

        x = self.conv1dLayer(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TwoD_CNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size):
        super(TwoD_CNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.main_period = main_period
        self.kernel_size = kernel_size
        
        self.conv2d = nn.Conv2d(input_size, input_size, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.conv1dLayer = Conv1dLayer(input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)      

    def forward(self, x):
        B, T, N = x.size()
        x = x.reshape(B, self.seq_length // self.main_period, self.main_period, N).permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, N)

        x = self.conv1dLayer(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
 
# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=9):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels//reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels//reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 2D-CNN-LSTM-Attention网络定义
class TwoD_CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size):
        super(TwoD_CNN_LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.main_period = main_period
        self.kernel_size = kernel_size
        
        self.conv2d = nn.Conv2d(input_size, input_size, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.ca1 = ChannelAttention(input_size)  # Channel Attention for the first convolution
        self.conv1dLayer = Conv1dLayer(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        B, T, N = x.size()
        x = x.reshape(B,  self.seq_length//self.main_period ,self.main_period, N).permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.ca1(x)  
        x = x.permute(0, 2, 3, 1).reshape(B, -1, N)
        x = self.conv1dLayer(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
class TwoD_CNN_GRU_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size):
        super(TwoD_CNN_GRU_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.main_period = main_period
        self.kernel_size = kernel_size
        
        self.conv2d = nn.Conv2d(input_size, input_size, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.ca1 = ChannelAttention(input_size)  # Channel Attention for the first convolution
        self.conv1dLayer = Conv1dLayer(input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        B, T, N = x.size()
        x = x.reshape(B, self.seq_length // self.main_period, self.main_period, N).permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.ca1(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, N)
        x = self.conv1dLayer(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def train_model(model_name, train_dataloader, val_dataloader):
    # 模型公有参数
    num_epochs = CONFIG[model_name]['num_epochs']
    learning_rate = CONFIG[model_name]['learning_rate']
    # 学习率衰减参数(不)
    lr_decay_step = 5
    lr_decay_gamma = 1
    # 早停参数
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0


    # CONFIG 是一个字典，包含所有模型的配置参数
    model_params = CONFIG[model_name]

    # 创建模型的通用参数
    # input_size = model_params['input_size']
    hidden_size = model_params['hidden_size']
    num_layers = model_params['num_layers']
    # output_size = CONFIG['COMMON']['output_size'] 
    # seq_length = model_params['seq_length']   

    # 特定参数
    kernel_size = model_params.get('kernel_size', None)  # 仅对 CNN_LSTM、TwoD_CNN_LSTM 有用
    main_period = model_params.get('main_period', None)  # 仅对 TwoD_CNN_LSTM 有用


    # 创建模型实例
    if model_name == 'TwoD_CNN_LSTM':
        model = TwoD_CNN_LSTM(input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size)
    elif model_name == 'TwoD_CNN_LSTM_Attention':
        model = TwoD_CNN_LSTM_Attention(input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size)
    elif model_name == 'CNN_LSTM':
        model = CNN_LSTM(input_size, hidden_size, num_layers, output_size, seq_length, kernel_size)
    elif model_name == 'LSTM':
        model = LSTM(input_size, hidden_size, num_layers, output_size, seq_length)
    elif model_name == 'CNN':
        model = CNN(input_size, hidden_size, num_layers, output_size, seq_length, kernel_size)
    elif model_name == 'GRU':
        model = GRU(input_size, hidden_size, num_layers, output_size, seq_length)
    elif model_name == 'CNN_GRU':
        model = CNN_GRU(input_size, hidden_size, num_layers, output_size, seq_length, kernel_size)
    elif model_name == 'TwoD_CNN_GRU':
        model = TwoD_CNN_GRU(input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size)
    elif model_name == 'TwoD_CNN_GRU_Attention':
        model = TwoD_CNN_GRU_Attention(input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size)

    model.to(device)


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    
    # 初始化最好的验证集损失
    best_val_loss = float('inf')
    # 记录train_loss和val_loss
    train_loss_list = []
    val_loss_list = []
    # 记录预测值和真实值
    predictions_train = []
    
    # 训练模型
    print("-------------------  Start training the model  -------------------")
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        train_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_loss_list.append(train_loss)
        
        # 学习率衰减
        scheduler.step()
        
        # 在验证集上进行评估
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        val_loss_list.append(val_loss)
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            torch.save(best_model_wts, os.path.join(save_data_path,f'model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print("Early stopping")
            break
    
    # 加载最好的模型权重
    # model.load_state_dict(best_model_wts)
    
    print("-------------------  Finish training the model  -------------------")
    print(f'Best Val Loss: {best_val_loss:.4f}')
    
    # 画出train_loss和val_loss
    plt.title(f"Train and Val loss of {model_name}")
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.legend()
    plt.savefig(os.path.join(save_data_path,'loss.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    return model


def create_targets(data, seq_length, pred_length):
        m = len(data) - seq_length - pred_length
        targets = np.zeros((m, pred_length))
        for i in range(m):
            x = data[i:i + seq_length]
            y = data[i + seq_length:i + seq_length + pred_length]
            targets[i] = y
        return targets

def test_model(model_name, test_dataloader, test_df_origin, model):
    # 将数据集转换为PyTorch的Tensor
    # test_data = torch.tensor(test_df, dtype=torch.float32).unsqueeze(1)

    # # 创建测试集的数据集对象
    # # 选取合适的seq_length和pred_length
    # # seq_length是用来预测的历史数据长度，pred_length是预测的未来数据长度
    # test_dataset = TimeSeriesDataset(test_data, seq_length, pred_length)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 加载已训练好的模型参数
    model.load_state_dict(torch.load(os.path.join(save_data_path,f'model.pt')))
    model.to(device)
    model.eval()
    # 在测试集上进行预测
    predictions = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            predictions.append(outputs.detach().cpu().numpy())

    # 创建真实值数据集
    test_true = create_targets(test_df_origin['Load'].values, seq_length, pred_length)

    # 将预测结果转换为pred_length维数组
    predictions = np.concatenate(predictions, axis=0).reshape(-1, 1)
    predictions = scaler.inverse_transform(np.concatenate((predictions,np.zeros((predictions.shape[0],8))),axis=1))[:,0].reshape(-1,output_size)
    np.save(os.path.join(save_data_path,'predition.npy'),predictions)
    np.save(os.path.join(save_data_path,'true.npy'),test_true)

    # print("predictions.shape",predictions.shape)
    # 画出预测结果
    # plt.figure(figsize=(10, 6))
    # 选择天数
    # n = 4223
    # n = 3800
    # plt.plot(test_true[n], label='True')
    # plt.plot(predictions[n], label='Predicted')
    # plt.legend()
    # plt.show(block=True)
    
    test_true = test_true.flatten()
    # 将预测结果转换为pred_length维数组
    predictions = predictions.flatten()

    # 画出预测结果
    # plt.figure(figsize=(10, 6))
    # plt.plot(test_true[-96:], label='True')
    # plt.plot(predictions[-96:], label='Predicted')
    # plt.legend()
    # plt.show(block=True)
    # plt.savefig(CONFIG['COMMON']['save_path']+'/image/实验图片/'+model_name+'_predition_'+str(n)+'.png')

    # # Scatter plot of the entire test set
    # plt.figure(figsize=(10, 6))
    # plt.scatter(test_true[::pred_length], predictions[::pred_length], alpha=0.5)
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title('Scatter plot of True vs Predicted Values')
    # plt.show()
    # plt.savefig(CONFIG['COMMON']['save_path']+'/image/实验图片/'+model_name+'_PreditionVsTrue.png')

    print('Metrics:')
    mae = mean_absolute_error(test_true, predictions)
    print(f'MAE: {mae:.2f}')
    rmse = sqrt(mean_squared_error(test_true, predictions))
    print(f'RMSE: {rmse:.2f}')
    # 计算MAPE
    mape = mean_absolute_percentage_error(test_true, predictions)*100
    print(f'MAPE: {mape:.2f}%')
    # 计算 R²
    r2 = r2_score(test_true, predictions)
    print(f'R²: {r2:.2f}')

def execute(model_name):  
    test_df_origin, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(model_name)
    model = train_model(model_name, train_dataloader, val_dataloader)
    test_model(model_name, test_dataloader, test_df_origin, model)

execute(model_name)