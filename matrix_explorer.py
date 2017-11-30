#!/usr/bin/python3

import json
from flask import Flask
app = Flask(__name__)
"""This program requires flask to be installed"""

langs = ["us", "es"]
models = {"bow": "bow_0",
          "char_cnn_lstm": "char_cnn_lstm_train_128_128_0",
          "lsvc": "lsvc_0"}
matrixs = {}
mappings = {}


def get_matrix_and_mapping(lang, model):
    if lang not in mappings:
        mapping = [x.split()[1:] for x in open("data/mappings/{}_mapping.txt".format(lang), "r").readlines()]
        mappings[lang] = mapping
    if (lang, model) not in matrixs:
        matrix = json.load(open("output/{}.{}.matrix.json".format(models[model], lang), "r"))
        matrixs[(lang, model)] = matrix
    return matrixs[(lang, model)], mappings[lang]


@app.route('/')
def index():
    s = "<html><head></head><body>"
    s += "<table><tr><td></td>"
    for lang in langs:
        s += "<th scope='col'>{}</th>".format(lang)
    s += "</tr>"
    for model in sorted(models.keys()):
        s += "<tr><th scope='row'>{}</th>".format(model)
        for lang in langs:
            s += "<td><a href='/{}/{}'>{}</a></td>".format(lang, model, "{} - {}".format(model, lang))
        s += "</tr>"
    s += "</table>"
    s += "</body></html>"
    return s


@app.route('/<lang>/<model>')
def hello_world(lang, model):
    if lang not in langs or model not in models:
        return "<html><head></head><body>Invalid url<br>Checkout valid configs at <a href='/'>at the index</a>.</body></html>"
    s = "<html><head></head><body>"
    s += "<table><tr><td></td>"
    try:
        matrix, mapping = get_matrix_and_mapping(lang, model)
    except:
        return "<html><head></head><body>Unfortunately the server couldn't load this dataset<br>Checkout other configs at <a href='/'>at the index</a>.</body></html>"
    dim = len(matrix)
    for i in range(dim):
        s += "<th scope='col'>{}</th>".format(mapping[i][0])
    s += "</tr>"
    for i in range(dim):
        s += "<tr><th scope='row'>{}</th>".format(mapping[i][0])
        for j in range(dim):
            s += "<td><a href='/text/{}/{}/{}/{}'>{}</a></td>".format(lang, model, i, j, len(matrix[i][j]))
        s += "</tr>"
    s += "</table>"
    s += "</body></html>"
    return s


@app.route('/text/<lang>/<model>/<int:gold>/<int:out>')
def get_text(lang, model, gold, out):
    if lang not in langs or model not in models:
        return "<html><head></head><body>Invalid url<br>Checkout valid configs at <a href='/'>at the index</a>.</body></html>"
    try:
        matrix, mapping = get_matrix_and_mapping(lang, model)
    except:
        return "<html><head></head><body>Unfortunately the server couldn't load this dataset<br>Checkout other configs at <a href='/'>at the index</a>.</body></html>"
    dim = len(matrix)
    s = '<html><head><link rel="stylesheet" href="/style"></head><body>'
    s += "<h1>Tweets that contained {} and labeled as {}</h1>".format(mapping[gold][0], mapping[out][0])
    s += '<div class="responsive"><table>'
    for data in matrix[gold][out]:
        if len(data) == 2:
            text, tokens = data
            s += "<tr><td>{}</td><td>{}</td></tr>".format(text, ",".join(tokens))
        else:
            text = data
            s += "<tr><td>{}</td><td>{}</td></tr>".format(text, "N/A")
    s += "</div></table></body></html>"
    return s  # "Gold: {} Out: {}\n".format(gold, out)+json.dumps(matrix[gold][out])


@app.route('/reload')
def reload():
    global matrixs, mappings
    matrixs = {}
    mappings = {}
    return "<html><head></head><body>Finished reload</body></html>"


@app.route('/style')
def style():
    return '''.responsive{display:block;overflow-x:auto}
table {
    border-collapse:collapse;border-spacing:0;width:100%;display:table;border:1px solid #ccc;
    table-layout:fixed;
}
table tr{border-bottom:1px solid #ddd}
table tr:nth-child(odd){background-color:#fff}
table tr:nth-child(even){background-color:#f1f1f1}
table td, table th {padding:8px 8px;display:table-cell;text-align:left;vertical-align:top;overflow-wrap: break-word;}
table th:first-child, table td:first-child{padding-left:16px}
'''


if __name__ == '__main__':
    app.run()
