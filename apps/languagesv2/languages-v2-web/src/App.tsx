import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import Quiz from './pages/Quiz';
import Nav from './pages/Nav';

function App() {
  return (
    <Router>
      <div className="App w-full flex flex-col items-center justify-center min-h-screen bg-gray-100">
        <Nav />
        <div className="w-4/5">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/quiz" element={<Quiz />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;