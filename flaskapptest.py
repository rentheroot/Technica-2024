from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Phoneme to IPA dictionary
phoneme_ipa_dict = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'ə', 'AY': 'ī', 'EH': 'ɛ',
    'ER': 'ɝ', 'EY': 'ā', 'IH': 'ɪ', 'IY': 'i', 'OW': 'ō', 'OY': 'ʉ', 'UH': 'ʊ',
    'UW': 'u', 'B': 'b', 'CH': 'ʧ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'ɡ',
    'HH': 'h', 'JH': 'ʤ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
    'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ', ' ': ''
}

@app.route('/')
def home():
    return render_template('keyboard.html', phoneme_ipa_dict=phoneme_ipa_dict)

@app.route('/key_pressed', methods=['POST'])
def key_pressed():
    symbol = request.json.get('symbol')
    print(f'Key pressed: {symbol}')  # Print to Python console
    return jsonify(success=True)

@app.route('/assets/<path:path>')
def send_report(path):
    return send_from_directory('assets', path)

if __name__ == '__main__':
    app.run(debug=True)
