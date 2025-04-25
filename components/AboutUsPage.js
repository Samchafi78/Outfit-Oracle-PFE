import React from 'react';
import '../styles/CommonPage.css';

function AboutUsPage() {
  return (
    <div className="page-container">
      <h1>À propos de nous</h1>
      <p>
        Nous sommes trois étudiants de <strong>Télécom SudParis</strong>, passionnés par l'intelligence artificielle et réseau informatique. 
        Notre projet <strong>OutfitOracle</strong> vise à proposer des recommandations vestimentaires intelligentes en combinant 
        l'apprentissage automatique et une interface utilisateur intuitive.
      </p>

      <h2>Notre Mission</h2>
      <p>
        Faciliter le choix des vêtements en fournissant des suggestions adaptées aux préférences et au style de chacun.
      </p>

      <h2>Notre Équipe</h2>
      <ul>
        <li><strong>CHAFI RAHAMATTOULLA Samir</strong> - Expert en développement web</li>
        <li><strong>WONG Hoe Ziet</strong> -UI/UX designer et intégration</li>
        <li><strong>ZHU Xingyu</strong> - Spécialiste en Machine Learning</li>
      </ul>

      <h2>Contact</h2>
      <p>Pour toute question, contactez-nous à <a href="mailto:contact@outfitoracle.com">contact@outfitoracle.com</a></p>
    </div>
  );
}

export default AboutUsPage;


