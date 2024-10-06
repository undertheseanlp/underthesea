import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import Nav from './pages/Nav';
import Quiz from './pages/Quiz';
import Video from './pages/Video';


function App() {
  return (
    <Router>
      <div className="App w-full flex flex-col items-center justify-center min-h-screen bg-gray-100">
        <Nav />
        <div className="w-4/5">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/quiz" element={<Quiz />} />
            <Route path="/video/:language" element={<Video />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;