import flask as fl
import numpy as np

app = fl.Flask(__name__)


@app.route('/')
def index():
   #return "hello"
    return app.send_static_file('index.html')

@app.route('/api/uniform')
def uniform():
    return {"Value":np.random.uniform()}


# Run in debug mode
if __name__ == "__main__":
   app.run(debug=True)