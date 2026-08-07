import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';

function DashboardPlaceholder() {
  return (
    <div className="flex-center" style={{ height: '100vh', color: 'var(--color-accent-cyan)' }}>
      <h1 className="font-display">DASHBOARD COMING SOON</h1>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<DashboardPlaceholder />} />
      </Routes>
    </Router>
  );
}

export default App;
