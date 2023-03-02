"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/4/19 17:56
@Email : handong_xu@163.com
"""
import xgboost as xgb
import numpy as np
from models.XGBoost import Model
from utils.dataloader import get_data
from utils.converter import dump

def main():
    model = Model()
    trainset = 'data/非本人.csv'
    X_train,X_test,y_train,y_test = get_data(trainset)
    clf = model.train(X_train,y_train,X_test,y_test)
    dump('ckpt/xgboost.trans_base.model',clf)
    pred = model.test(X_test,clf)
    model.evaluate(y_test,pred)



if __name__ == '__main__':
    main()
