from flask import Flask, render_template, request, redirect, url_for
from image_generator.table_generator import *

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        timesteps = int(request.form["timesteps"])
        num_images = int(request.form["num_images_per_prompt"])

        return redirect(url_for('display', prompt=prompt, timesteps=timesteps, num_images=num_images))
    return render_template("index.html")


@app.route("/display")
def display():

    prompt = request.args.get('prompt')
    timesteps = int(request.args.get('timesteps'))
    num_images = int(request.args.get('num_images'))

    images = generate_images(prompt, timesteps, num_images)

    if images:
        image_data = [pil_to_base64(img) for img in images]
        return render_template("display.html", images=image_data)
    else:
        return "Error generating images"


if __name__ == "__main__":
    app.run(debug=True)
