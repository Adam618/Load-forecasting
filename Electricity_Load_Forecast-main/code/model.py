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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_name = 'CNN_LSTM'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = StandardScaler()
CONFIG = func.load_config()
func.print_config(model_name)
data_path = CONFIG['COMMON']['data_path']  
seq_length = CONFIG['COMMON']['seq_length']
pred_length = CONFIG['COMMON']['pred_length']
output_size = CONFIG['COMMON']['pred_length']
save_model_path = CONFIG['COMMON']['save_path']+'/model/best_'+model_name+'.pt'
save_image_path = CONFIG['COMMON']['save_path']+'/image/实验图片/'+model_name+'_loss.png'



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_length, hidden_size]
        attention_scores = self.attention_weights(lstm_output)  # [batch_size, seq_length, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_length, 1]
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # [batch_size, hidden_size]
        return context_vector


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

def FFT_for_Period(x, k=2):
    # [B, T, C]
    x = x[:,:,0].unsqueeze(2)
    # print('x.shape',x.shape)
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.main_period = main_period
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv2d(input_size, input_size, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        # self.conv2 = nn.Conv2d(input_size, input_size, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, x):
        B, T, N = x.size()
        x = x.reshape(B,  self.seq_length//self.main_period ,self.main_period,
                              N).permute(0, 3, 1, 2).contiguous()
        # x = x.reshape(B, self.main_period, self.seq_length//self.main_period ,
        #                       N).permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, N)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    # def forward(self, x):
    #     B, T, N = x.size()
    #     x = x.reshape(B, self.seq_length // self.main_period, self.main_period, N).permute(0, 3, 1, 2).contiguous()
    #     x = self.conv1(x)
    #     x = x.permute(0, 2, 3, 1).reshape(B, -1, N)
    #     h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    #     c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    #     lstm_out, _ = self.lstm(x, (h0, c0))
    #     context_vector = self.attention(lstm_out)
    #     out = self.fc(context_vector)
    #     return out
    

def get_dataloaders(model_name):# 读取训练集和验证集的数据
    batch_size = CONFIG[model_name]['batch_size']

    df = pd.read_csv(data_path)
    m_all = len(df) # 数据集总行数
    m_val = 52*96 # 验证集数量
    m_test = 52*96 # 测试集数量
    m_train = m_all - m_test - m_val # 训练集数量

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
    scaler.fit(train_df)
    train_df = scaler.transform(train_df)
    # 对验证集和测试集进行相同的归一化操作
    val_df = scaler.transform(val_df)
    test_df_origin = test_df
    test_df = scaler.transform(test_df)
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

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

def train_model(model_name, train_dataloader, val_dataloader):
    # 模型公有参数
    num_epochs = CONFIG[model_name]['num_epochs']
    learning_rate = CONFIG[model_name]['learning_rate']
    # 学习率衰减参数(不)
    lr_decay_step = 15
    lr_decay_gamma = 1
    # 早停参数
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    # 模型私有参数
    if model_name == 'CNN_LSTM':
        input_size = CONFIG['CNN_LSTM']['input_size']
        hidden_size = CONFIG['CNN_LSTM']['hidden_size']
        num_layers = CONFIG['CNN_LSTM']['num_layers']
        main_period = CONFIG['CNN_LSTM']['main_period']
        kernel_size = CONFIG['CNN_LSTM']['kernel_size']
        model = CNN_LSTM(input_size, hidden_size, num_layers, output_size, seq_length, main_period, kernel_size)
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
            save_model_path
            torch.save(best_model_wts, save_model_path)
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
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.legend()
    plt.savefig(save_image_path, dpi=600, bbox_inches='tight')
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
    model.load_state_dict(torch.load(save_model_path))
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
    # 画出预测结果
    plt.figure(figsize=(10, 6))
    # 选择天数
    n = 4223
    plt.plot(test_true[n], label='True')
    plt.plot(predictions[n], label='Predicted')
    plt.legend()
    plt.show()

    test_true = test_true.flatten()
    # 将预测结果转换为pred_length维数组
    predictions = predictions.flatten()

    # 画出预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(test_true[-96:], label='True')
    plt.plot(predictions[-96:], label='Predicted')
    plt.legend()
    plt.show()
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