import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import axios from "axios";
import "../styles/Dashboard.css";

const Dashboard = () => {
  const [analytics, setAnalytics] = useState({
    totalUploads: 0,
    totalUsers: 0,
    uploadsByDay: []
  });

  const [logs, setLogs] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:5000/api/dashboard")
      .then(response => setAnalytics(response.data))
      .catch(error => console.error("Erreur lors de la récupération des données: ", error));

    axios.get("http://localhost:5000/api/logs")
      .then(response => setLogs(response.data))
      .catch(error => console.error("Erreur lors de la récupération des logs: ", error));
  }, []);

  return (
    <div className="dashboard">
      <h1>📊 Tableau de Bord</h1>

      <div className="stats">
        <div className="stat-card">📸 Images Uploadées: {analytics.totalUploads}</div>
        <div className="stat-card">👥 Utilisateurs: {analytics.totalUsers}</div>
      </div>

      <div className="charts">
        <div className="chart-container">
          <h2>📅 Nombre d'Uploads par Jour (1 Mois)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={analytics.uploadsByDay || []}>
              <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString()} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="uploads" stroke="#0088FE" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Cadre pour afficher les logs */}
      <div className="logs-container">
        <h2>📜 Logs des dernières actions</h2>
        <div className="logs-list">
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <div key={index} className="log-entry">
                <p><strong>Date:</strong> {new Date(log.date).toLocaleString()}</p>
                <p><strong>Image:</strong> {log.image}</p>
                <p><strong>Âge:</strong> {log.age}</p>
                <p><strong>Budget:</strong> {log.budget} €</p>
              </div>
            ))
          ) : (
            <p>Aucune action récente.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
