import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div className="w-full pt-24 space-y-4">
      {/* Header Section */}
      <header className="text-left">
        <h1 className="text-5xl font-bold mb-2 text-blue-700">Languages v2</h1>
        <p className="text-xl text-gray-700">Like Duolingo, but Opensource</p>
      </header>

      {/* English Section */}
      <section className="bg-white shadow-md rounded-lg p-6 transition-transform transform hover:scale-105 hover:bg-blue-50">
        <h2 className="text-3xl font-semibold mb-4 text-blue-700">English 🇬🇧</h2>
        <p className="text-md text-gray-700 mb-6">Master English with our engaging quizzes and exercises. Unlock the full potential of your language skills and take your communication to the next level!</p>
        <div className="mb-8 space-x-4">
          <Link
            to="/quiz?language=English"
            className="inline-block bg-blue-600 text-white px-6 py-3 rounded-full font-semibold text-center hover:bg-blue-800 shadow-lg">
            🚀 Start Your English Adventure Now!
          </Link>
          <Link
            to="/video/English"
            className="inline-block bg-blue-400 text-white px-6 py-3 rounded-full font-semibold text-center hover:bg-blue-600 shadow-lg">
            🎥 Watch English Videos
          </Link>
        </div>
      </section>

      {/* Vietnamese Section */}
      <section className="bg-white shadow-md rounded-lg p-6 transition-transform transform hover:scale-105 hover:bg-green-50">
        <h2 className="text-3xl font-semibold mb-4 text-green-700">Vietnamese 🇻🇳</h2>
        <p className="text-md text-gray-700 mb-6">Discover the beauty of the Vietnamese language through fun and immersive quizzes. Let’s explore the culture and language together!</p>
        <div className="mb-8 space-x-4">
          <Link
            to="/quiz?language=Vietnamese"
            className="inline-block bg-green-600 text-white px-6 py-3 rounded-full font-semibold text-center hover:bg-green-800 shadow-lg">
            🌟 Start Vietnamese Quiz and Uncover the Magic!
          </Link>
          <Link
            to="/video/Vietnamese"
            className="inline-block bg-green-400 text-white px-6 py-3 rounded-full font-semibold text-center hover:bg-green-600 shadow-lg">
            🎥 Watch Vietnamese Videos
          </Link>
        </div>
      </section>

      {/* Chinese Section */}
      <section className="bg-white shadow-md rounded-lg p-6 transition-transform transform hover:scale-105 hover:bg-red-50">
        <h2 className="text-3xl font-semibold mb-4 text-red-700">Chinese 🇨🇳</h2>
        <p className="text-md text-gray-700 mb-6">Begin your Mandarin Chinese journey with fun, engaging quizzes. Unlock the wonders of the Chinese language and culture!</p>
        <div className="mb-8 space-x-4">
          <Link
            to="/quiz?language=Chinese"
            className="inline-block bg-red-600 text-white px-6 py-3 rounded-full font-semibold text-center hover:bg-red-800 shadow-lg">
            🐉 Start Chinese Quiz and Begin Your Journey!
          </Link>
          <Link
            to="/video/Chinese"
            className="inline-block bg-red-400 text-white px-6 py-3 rounded-full font-semibold text-center hover:bg-red-600 shadow-lg">
            🎥 Watch Chinese Videos
          </Link>
        </div>
      </section>

      {/* Footer Section */}
      <footer className="text-center text-gray-600 mt-16">
        <p className="text-lg">Happy learning! Stay consistent, stay curious. 🚀 Let every quiz be a new adventure!</p>
      </footer>
    </div>
  );
}

export default Home;