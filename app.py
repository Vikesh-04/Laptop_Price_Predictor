from flask import Flask, request, jsonify, render_template
import numpy as np
import pycaret
from pycaret.regression import *
app = Flask(__name__, template_folder='templates')

model = load_model('final_blend_top5')
cols=['product_name','product_processor_core','product_processor_gen','product_storage_type','product_hdd','product_ssd','product_ram','product_dedicated_graphics','product_graphics_ram','product_touchscreen','product_display_size']

@app.route('/')
def main():
    return render_template('main.html')


@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final_features = np.array(features)
    data=pd.DataFrame([final_features],columns=cols)
    prediction=predict_model(model,data=data)
    prediction=prediction.Label[0]
    return render_template('main.html', prediction_text='Laptop price should be around Rs. {}'.format(round(prediction)))

if __name__ == '__main__':
	app.run()