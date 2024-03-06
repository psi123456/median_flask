from flask import Flask, jsonify
from ultralytics import Explorer
from ultralytics.data.explorer import plot_query_result
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Explorer 객체 초기화
exp = Explorer(data='C:\\team\\dataset14.yaml', model='C:\\team\\best.pt')
exp.create_embeddings_table()


@app.route('/get-images', methods=['GET'])
def get_images():
    df = exp.ask_ai("알약이 있는사진 10장 출력해줘")
    if df.empty:
        return jsonify({"message": "검색 결과가 없습니다."})
    else:
        fig, ax = plt.subplots()
        plot_query_result(df, ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return jsonify({"image": image_base64})

if __name__ == '__main__':
    app.run(debug=True)
