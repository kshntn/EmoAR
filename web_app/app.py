import os
from flask import Flask, request, render_template
app = Flask(__name__)

from inference import get_flower_name


@app.route('/', methods=['GET', 'POST'])
def hello_world():
	if request.method == 'GET':
		return render_template('index.html', value="hi hi mno")
	if request.method == 'POST':
		if 'file' not in request.files:
			print("file not uploaded")
			return
		file = request.files['file']
		image=file.read()
		category, flower_name = get_flower_name(image_bytes=image)
		return render_template('result.html', flower =flower_name, category=category)

if __name__ == '__main__':
   	app.run(debug=True, port=os.getenv('PORT', 5000))
