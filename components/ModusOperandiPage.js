import React from "react";
import "../styles/CommonPage.css";

function ModusOperandiPage() {
  return (
    <div className="page-container">
      <h1>Modus Operandi</h1>
      <p>
        Bienvenue sur Outfit Oracle, votre assistant intelligent en recommandations de mode. 
        Notre plateforme utilise un algorithme avancé pour analyser vos préférences et vous proposer des tenues adaptées à votre style et budget.
      </p>
      
      <h2>Comment ça marche ?</h2>
      <ol>
        <li>Importez une image d’un vêtement ou sélectionnez vos préférences.</li>
        <li>Indiquez votre âge et votre budget.</li>
        <li>Notre système analyse vos choix et propose des recommandations personnalisées.</li>
        <li>Parcourez les suggestions et trouvez l’ensemble parfait pour vous !</li>
      </ol>
      
      <h2>Pourquoi utiliser Outfit Oracle ?</h2>
      <ul>
        <li>Suggestions basées sur vos goûts et tendances actuelles.</li>
        <li>Interface simple et intuitive.</li>
        <li>Optimisation du style avec des recommandations adaptées.</li>
      </ul>
      <h2>Technologies utilisées</h2>
        <p>
          Cette application est propulsée par deux technologies de l’IA, notamment le Machine Learning et le Deep Learning. 
          Le Deep Learning est utilisé pour reconnaître l’image donnée en entrée et en extraire les caractéristiques visuelles grâce à la vision par ordinateur. 
          Quant à la partie Machine Learning, elle permet de recommander l’article le plus similaire ainsi que d’autres articles complémentaires pour composer un look complet. 
          Nous utilisons une approche de type <em>User-to-User Collaborative Filtering</em> pour affiner les recommandations selon les préférences utilisateur.
        </p>
    </div>
  );
}

export default ModusOperandiPage;
