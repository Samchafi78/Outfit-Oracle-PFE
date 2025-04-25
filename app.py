from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import datetime
import os
import logging
import subprocess
import json
from flask import send_from_directory


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
CORS(app)

# Configuration de la base de données MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://samir:samir@localhost/pfe_bd'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuration du dossier d'upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crée le dossier s'il n'existe pas
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db = SQLAlchemy(app)

# Modèle de base de données
class UserInput(db.Model):
    __tablename__ = "user_input"  
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255))
    age = db.Column(db.Integer)
    budget = db.Column(db.Float)
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())

# Route pour récupérer les statistiques du tableau de bord
@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    today = datetime.date.today()
    last_month = today - datetime.timedelta(days=30)

    total_uploads = UserInput.query.count()
    total_users = db.session.query(UserInput.age).distinct().count()

    # Nombre d'uploads par jour (30 derniers jours)
    uploads_by_day = (
        db.session.query(
            func.date(UserInput.created_at).label("date"),
            func.count(UserInput.id).label("uploads")
        )
        .filter(UserInput.created_at >= last_month)
        .group_by(func.date(UserInput.created_at))
        .order_by(func.date(UserInput.created_at))
        .all()
    )
    uploads_by_day_data = [{"date": str(day), "uploads": count} for day, count in uploads_by_day]

    return jsonify({
        "totalUploads": total_uploads,
        "totalUsers": total_users,
        "uploadsByDay": uploads_by_day_data
    })

# Route pour récupérer les logs des uploads
@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs = (
        db.session.query(UserInput.created_at, UserInput.image_path, UserInput.age, UserInput.budget)
        .order_by(UserInput.created_at.desc())  # Trier du plus récent au plus ancien
        .limit(50)  # Récupérer les 50 dernières actions
        .all()
    )

    logs_data = [
        {
            "date": str(log[0]),
            "image": log[1],
            "age": log[2],
            "budget": log[3]
        } 
        for log in logs
    ]

    return jsonify(logs_data)

# Route pour stocker les données (avec upload d'image)
@app.route('/api/store_data', methods=['POST'])
def store_data():
    print("Requête reçue:", request.files, request.form)

    if "image" not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400

    file = request.files["image"]
    age = request.form.get("age")
    budget = request.form.get("budget")

    if not age or not budget:
        return jsonify({"error": "L'âge et le budget sont obligatoires"}), 400
    
    
    # Enregistrer le fichier temporairement
    filename = os.path.join("temp", file.filename)  # Dossier temporaire
    os.makedirs("temp", exist_ok=True)
    file.save(filename)
    
    

    # Enregistrement dans la base de données
    new_entry = UserInput(image_path=filename, age=int(age), budget=float(budget))
    db.session.add(new_entry)
    db.session.commit()
    
    # Lancer ML_local.py et récupérer les résultats
    try:
        subprocess.run(["python", "PFE_model/ML_local.py", filename], check=True)

        # Lire le fichier JSON de sortie
        output_path = os.path.join("PFE_model", "output", "results.json")
        with open(output_path, "r") as f:
            results = json.load(f)

        print("✅ Résultats retournés :", results)
        return jsonify(results), 201

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Erreur lors de l'exécution du modèle : {e}"}), 500

@app.route('/api/get_results', methods=['GET'])
def get_results():
    try:
        results_path = os.path.join("PFE_model", "output", "results.json")
        with open(results_path, "r") as f:
            results = json.load(f)
        print("results:",results)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/images/images/<folder>/<filename>')
def serve_image(folder, filename):
    return send_from_directory(os.path.join("images", "images", folder), filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
