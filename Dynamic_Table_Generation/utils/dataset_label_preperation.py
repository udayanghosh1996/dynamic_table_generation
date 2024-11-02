import os
import json
from html import escape
from bs4 import BeautifulSoup as bs
from jsonl2json import JsonlToJsonFormatter

os.chdir('F:\\MTech_IIT_Jodhpur\\MTP')
os.getcwd()

path = './Dataset/PubTabNet/pubtabnet/pubtabnet/PubTabNet_2_0_0.jsonl'
path1 = './Dataset/PubTabNet/pubtabnet/pubtabnet/PubTabNet_2_0_0.json'
train_json = './Dataset/PubTabNet/pubtabnet/pubtabnet/PubTabNet_2_0_0_train.json'
val_json = './Dataset/PubTabNet/pubtabnet/pubtabnet/PubTabNet_2_0_0_val.json'


def train_val_label_seperation():
    jsonl = JsonlToJsonFormatter(os.path.join(os.getcwd(), path), os.path.join(os.getcwd(), path1))
    jsonl.to_json()
    with open(os.path.join(os.getcwd(), path1), encoding="utf8") as f:
        data = json.load(f)

    train = []
    val = []

    for datapoint in data:
        if datapoint['split'] == 'train':
            train.append(datapoint)
        elif datapoint['split'] == 'val':
            val.append(datapoint)

    with open(train_json, "w") as train_file:
        json.dump(train, train_file)

    with open(val_json, "w") as val_file:
        json.dump(val, val_file)
    return


def format_html(label):
    # Formats HTML code from tokenized annotation of label
    html_code = label['html']['structure']['tokens'].copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], label['html']['cells'][::-1]):
        if cell['tokens']:
            cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)
    html_code = ''.join(html_code)
    html_code = '''<html>
                   <head>
                   <meta charset="UTF-8">
                   <style>
                   table, th, td {
                     border: 1px solid black;
                     font-size: 10px;
                   }
                   </style>
                   </head>
                   <body>
                   <table frame="hsides" rules="groups" width="100%%">
                     %s
                   </table>
                   </body>
                   </html>''' % html_code

    # prettify the html
    soup = bs(html_code)
    html_code = soup.prettify()
    return html_code


def create_useful_label(file):
    target_filename = file.replace('.json', '_final.json')
    labels = []
    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)
    for occ in data:
        dic = {}
        dic['imgid'] = occ['imgid']
        dic['html'] = format_html(occ)
        dic['filename'] = occ['filename']
        labels.append(dic)
    with open(target_filename, "w") as final_file:
        json.dump(labels, final_file)
    return


if __name__=="__main__":

    train_val_label_seperation()
    create_useful_label(os.path.join(os.getcwd(), val_json))
    create_useful_label(os.path.join(os.getcwd(), train_json))