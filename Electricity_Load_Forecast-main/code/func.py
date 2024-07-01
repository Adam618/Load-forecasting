'''
Author: Adam618 cy5115236@163.com
Date: 2024-06-13 15:14:00
LastEditors: Adam618 cy5115236@163.com
LastEditTime: 2024-06-13 18:33:50
FilePath: /6.5_电力负荷预测/Electricity_Load_Forecast-main/code/func.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
#!/usr/bin/env python
# coding: utf-8

import os
from datetime import datetime
import json
import pandas as pd
import pprint
from tabulate import tabulate

def load_config(file_path="config_96.json"):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

def print_config(file_path="config_96.json"):
    with open(file_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
        
        common_params = config.get('COMMON', {})
        model_name = common_params.get('model_name', {})
        model_params = config.get(model_name, {})
        
        # Convert dictionaries to list of tuples for tabulate
        common_table = [(k, v) for k, v in common_params.items()]
        model_table = [(k, v) for k, v in model_params.items()]
        
        print("Common Parameters:")
        print(tabulate(common_table, headers=["Parameter", "Value"], tablefmt="pretty"))
        
        print(f"\n{model_name} Parameters:")
        print(tabulate(model_table, headers=["Parameter", "Value"], tablefmt="pretty"))