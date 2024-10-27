from flask import Flask, render_template, request, jsonify, send_from_directory
import model_maker
app = Flask(__name__)
global model_build
global live_predict
global user_input
user_input = 'ʃʊɹ '
model_build = model_maker.Build_Model(ipa_text = None, model_is_saved = True)
live_predict = model_maker.Predictor(model_build.model, model_build.vectorize_layer)



# Phoneme to IPA dictionary
phoneme_ipa_dict = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'ə', 'AY': 'ī', 'EH': 'ɛ',
    'ER': 'ɝ', 'EY': 'ā', 'IH': 'ɪ', 'IY': 'i', 'OW': 'ō', 'OY': 'ʉ', 'UH': 'ʊ',
    'UW': 'u', 'B': 'b', 'CH': 'ʧ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'ɡ',
    'HH': 'h', 'JH': 'ʤ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
    'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ', ' ': ' '
}

@app.route('/')
def home():
    global user_input
    return render_template('keyboard.html', phoneme_ipa_dict=phoneme_ipa_dict, user_input=user_input)

@app.route('/key_pressed', methods=['POST'])
def key_pressed():
    global user_input
    global model_build
    global live_predict
    global predictions
    symbol = request.json.get('symbol')
    print(f'Key pressed: {symbol}')  # Print to Python console
    user_input = user_input + phoneme_ipa_dict[symbol]
    print(user_input)
    if len(user_input) > 2:
        if len(user_input) > 20:
            user_input = user_input[-20:]
        
        if user_input.strip(): 
            predictions = live_predict.predict_user_input_with_prefix(user_input, top_k=6)
        print("Top predictions:", predictions)
    return jsonify(success=True)

@app.route('/input', methods=['GET'])
def handle_user_input():
    global predictions
    return render_template('in.html', user_input=predictions)

@app.route('/assets/<path:path>')
def send_report(path):
    return send_from_directory('assets', path)

if __name__ == '__main__':
    app.run(debug=True)
