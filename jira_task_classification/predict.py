#!/usr/bin/python

import data.make_dataset as preprocess
import sys
print(sys.path)

from models.predict_model import predict_task

if __name__ == '__main__':
    text_input = 'prepare for interview'
    data = {}
    data['task'] ='EXSCHOEN-1'
    data['summary'] = preprocess.text_preprocessing(text_input)

    predicted = predict_task(data)

