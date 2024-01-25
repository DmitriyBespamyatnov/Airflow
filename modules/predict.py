import dill
import json
import pandas as pd

list_json = ['7310993818.json', '7313922964.json', '7315173150.json', '7316152972.json', '7316509996.json']


def load_pkl(path):
    with open(path, 'rb') as mod:
        return dill.load(mod)


def prediction(model):
    dict_predict = {'json': [], 'predict': []}

    for json_el in list_json:
        with open(f'../data/test/{json_el}') as file:
            j = json.load(file)
            pred = model.predict(pd.DataFrame.from_dict([j]))
            dict_predict['json'].append(json_el)
            dict_predict['predict'].append(pred[0])
    return pd.DataFrame.from_dict(dict_predict)


def load_to_csv(df):
    df.to_csv('../data/predictions/predict.csv')


def predict():
    load_to_csv(prediction(load_pkl('../data/models/cars_pipe.pkl')))


if __name__ == '__main__':
    predict()
