from flask import Flask, jsonify, request, send_from_directory
import os

app = Flask(__name__, static_folder="frontend/public", static_url_path="/")

# Пример API-эндпоинта (проверка здоровья)
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)