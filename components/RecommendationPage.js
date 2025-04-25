import React, { useState } from "react";
import "../styles/RecommendationPage.css";

function RecommendationPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [age, setAge] = useState("");
  const [budget, setBudget] = useState("");
  const [bestArticles, setBestArticles] = useState({});
  const [recommendedImages, setRecommendedImages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);  // ✅ Stocke le fichier réel
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setAge("");
    setBudget("");
    setRecommendedImages([]);
    setBestArticles([]);
  };

  const handleConfirm = async () => {
    if (!selectedImage || !age || !budget) {
      alert("Veuillez remplir tous les champs !");
      return;
    }

    // 🧹 Réinitialise les images affichées à chaque clic
    setRecommendedImages([]);
    setBestArticles([]);

    const formData = new FormData();
    formData.append("image", selectedImage, selectedImage.name);
    formData.append("age", age);
    formData.append("budget", budget);
    
    console.log("📡 Envoi des données :", formData);
    setLoading(true); // ⏳ Démarre le chargement

    try {
      const storeResponse = await fetch("http://127.0.0.1:5000/api/store_data", {
        method: "POST",
        body: formData,
      });
  
      if (!storeResponse.ok) {
        const error = await storeResponse.json();
        alert("Erreur lors de l'enregistrement : " + error.error);
        setLoading(false);
        return;
      }
  
      const result = await storeResponse.json();

      // 📌 🔼 Best articles
      if (result.best_articles) {
        setBestArticles(result.best_articles);
      }
  
      if (result.recommendations && result.recommendations.length > 0) {
        const basePath = "http://127.0.0.1:5000/static/images/images";
  
        const imageElements = result.recommendations.map(id => {
          const paddedId = id.padStart(10, '0'); // Remet les 0 à gauche
          const folder = paddedId.substring(0, 3);
          const imageUrl = `${basePath}/${folder}/${paddedId}.jpg`;
          return (
            <img
              key={id}
              src={imageUrl}
              alt={`Article ${id}`}
              className="recommended-image"
            />
          );
        });
  
        setRecommendedImages(imageElements);
      } else {
        alert("Aucune recommandation trouvée.");
      }
    } catch (error) {
      console.error("Erreur :", error);
      alert("Une erreur est survenue.");
    } finally {
      setLoading(false); // ✅ Fin du chargement
    }

  };

  return (
    <div className="recommendation-page">
      <h1>Importez une Image pour une Recommandation</h1>
  
      <input 
        type="file" 
        accept="image/*" 
        onChange={handleImageChange} 
        className="file-input"
      />
  
      <input
        type="number"
        placeholder="Âge"
        value={age}
        onChange={(e) => setAge(e.target.value)}
        className="input-field"
      />
  
      <input
        type="number"
        placeholder="Budget (€)"
        value={budget}
        onChange={(e) => setBudget(e.target.value)}
        className="input-field"
      />
  
      {selectedImage && (
        <div className="image-preview">
          <h2>Aperçu :</h2>
          <img src={URL.createObjectURL(selectedImage)} alt="Prévisualisation" />
        </div>
      )}
  
      <button onClick={handleConfirm} className="confirm-button">Confirmer</button>
      <button onClick={handleReset} className="reset-button">Réinitialiser</button>
      
      {loading && (
        <div className="loading-indicator">
          <p>🤖 L'IA travaille sur vos recommandations...</p>
          <div className="spinner"></div>
        </div>
      )}
      
      {Object.keys(bestArticles).length > 0 && (
  <div className="best-articles-section">
    <h2>Meilleurs articles choisis par l'IA :</h2>
    <div className="image-grid">
      {["upper", "lower", "outer"].map(category => {
        const id = bestArticles[category];
        const paddedId = id.padStart(10, '0');
        const folder = paddedId.substring(0, 3);
        const imageUrl = `http://127.0.0.1:5000/static/images/images/${folder}/${paddedId}.jpg`;

        return (
          <div key={category} className="best-article-block">
            <h3>{category.toUpperCase()}</h3>
            <img
              src={imageUrl}
              alt={`${category} article`}
              className="recommended-image"
            />
          </div>
        );
      })}
    </div>
  </div>
)}
      
      
      {/* 🔽 Affichage des recommandations IA */}
      {recommendedImages.length > 0 && (
        <div className="recommendations-section">
          <h2>Articles recommandés :</h2>
          <div className="image-grid">
            {recommendedImages}
          </div>
        </div>
      )}
    </div>
  );
}

export default RecommendationPage;
