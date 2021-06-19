#!/usr/bin/python

import data.make_dataset as preprocess
import models.predict_model as model

if __name__ == '__main__':
    text_input = 'prepare for interview'
    data = {}
    data['task'] ='EXSCHOEN-1'
    data['summary'] = preprocess.text_preprocessing(text_input)

    predicted = model.predict_task(data)

