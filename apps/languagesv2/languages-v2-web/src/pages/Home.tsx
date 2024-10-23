import React, { useState } from 'react';
import VietnameseWords, { VietnameseWord } from './VietnameseWords';

// Example build version, replace this with an environment variable if needed.
const buildVersion = '1.0.0';

const frequentVietnameseWords = VietnameseWords;

function Home() {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoveredType, setHoveredType] = useState<string | null>(null);

  const playSound = (index: number, type: string) => {
    const audio = new Audio(`/audio/word-${index}-${type}.mp3`);
    audio.play();
  };

  const handleMouseEnter = (index: number, type: string) => {
    setHoveredIndex(index);
    setHoveredType(type);
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
    setHoveredType(null);
  };

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-indigo-500 via-purple-400 to-pink-300 pt-24">
      <div className="max-w-6xl mx-auto px-6 py-10">
        <h1 className="text-4xl font-extrabold text-center text-white mb-12 drop-shadow-lg">
          Frequent Vietnamese Words
        </h1>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-8">
          {frequentVietnameseWords.map((wordItem: VietnameseWord, index: number) => {
            const { word, partOfSpeech, frequency } = wordItem;

            return (
              <div
                key={index}
                className="relative p-6 bg-white bg-opacity-80 rounded-xl shadow-md transition-transform transform hover:scale-105 hover:shadow-xl hover:bg-opacity-100"
              >
                <div
                  className="absolute top-0 left-0 h-2 rounded-t-xl bg-gradient-to-r from-green-400 via-blue-500 to-indigo-600 shadow-md"
                  style={{
                    width: `${Math.min(frequency * 10, 100)}%`,
                    transition: 'width 0.3s ease-in-out',
                    boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.2)',
                  }}
                ></div>
                <h2 className="text-2xl font-bold text-indigo-800 mt-2">{word}</h2>
                <p className="italic text-gray-500 mb-4">{partOfSpeech}</p>
                <div className="flex justify-center space-x-6 mt-4">
                  <button
                    onClick={() => playSound(index, 'pronunciation')}
                    onMouseEnter={() => handleMouseEnter(index, 'pronunciation')}
                    onMouseLeave={handleMouseLeave}
                    className="text-indigo-600 hover:text-indigo-800 focus:outline-none transition-colors transform hover:scale-110 duration-300"
                    aria-label="North Pronunciation"
                  >
                    {hoveredIndex === index && hoveredType === 'pronunciation' ? (
                      <span className="bg-indigo-600 text-white px-2 py-1 rounded-lg transition-opacity duration-300 ease-in-out">
                        NTH
                      </span>
                    ) : (
                      <i className="fa fa-volume-up text-xl"></i>
                    )}
                  </button>
                  <button
                    onClick={() => playSound(index, 'example')}
                    onMouseEnter={() => handleMouseEnter(index, 'example')}
                    onMouseLeave={handleMouseLeave}
                    className="text-pink-600 hover:text-pink-800 focus:outline-none transition-colors transform hover:scale-110 duration-300"
                    aria-label="South Pronunciation"
                  >
                    {hoveredIndex === index && hoveredType === 'example' ? (
                      <span className="bg-pink-600 text-white px-2 py-1 rounded-lg transition-opacity duration-300 ease-in-out">
                        STH
                      </span>
                    ) : (
                      <i className="fa fa-volume-up text-xl"></i>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <footer className="mt-20 bg-white bg-opacity-70 backdrop-blur-md py-6">
        <div className="text-center text-gray-800 text-sm">
          Made with <span className="text-red-500 animate-pulse">❤️</span> by <strong className="text-indigo-800">Thảnh Thơi Team</strong>
        </div>
        <div className="text-center text-gray-600 mt-2">
          Build Version: <span className="font-semibold">{buildVersion}</span>
        </div>
      </footer>
    </div>
  );
}

export default Home;