import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Navbar.css';

function Navbar() {
  return (
    <header className="navbar">
      <div className="logo-container">
        {/* Logo du site - Redirige vers la page d'accueil */}
        <Link to="/">
          <img src={`${process.env.PUBLIC_URL}/OutfitOracle.png`} alt="Logo Site" className="logo" />
        </Link>

        {/* Logo TSP - Redirige vers le site de Télécom SudParis */}
        <a href="https://www.telecom-sudparis.eu/" target="_blank" rel="noopener noreferrer">
          <img src={`${process.env.PUBLIC_URL}/logo_tsp.png`} alt="Logo TSP" className="logo" />
        </a>
      </div>
      <nav className="nav-links">
        <Link to="/">Accueil</Link>
        <Link to="/recommendation">Recommandation</Link>
        <Link to="/modus_operandi">Modus Operandi</Link>
        <Link to="/about_us">About Us</Link>
      </nav>
    </header>
  );
}

export default Navbar;
