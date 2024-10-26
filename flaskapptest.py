from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('keyboard.html')

@app.route('/key_pressed', methods=['POST'])
def key_pressed():
    symbol = request.json.get('symbol')
    print(f'Key pressed: {symbol}')  # Print to Python console
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
