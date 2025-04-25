import React from 'react';
import '../styles/HomePage.css';
import { Link } from "react-router-dom";

function HomePage() {
  return (
    <div className="homepage">
      <div className="landscape-item">
        <img src={`${process.env.PUBLIC_URL}/landscape.jpg`} alt="Landscape" />
      </div>

      <div className="cta-section">
        <Link to="/recommendation"><button className="try-now-button">Try Now</button></Link>
      </div>
    </div>
  );
}

export default HomePage;
