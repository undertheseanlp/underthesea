import React, { useState } from 'react';
import VietnameseWords, { VietnameseWord } from './VietnameseWords';
import { BUILD_VERSION, DEVELOPMENT_TEAM } from './Config';
import Audios from './AudiosData';

const frequentVietnameseWords = VietnameseWords;

function Home() {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoveredType, setHoveredType] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>('');

  const playSound = (word: VietnameseWord, speaker_id: number) => {
    const wordItem = Audios.find((audio) => audio.word === word.word && audio.speaker_id === speaker_id);
    if (!wordItem) {
      console.warn('Audio not found for the given word and speaker');
      return;
    }

    const audioUrl = `https://undertheseanlp.com/data/audios/${wordItem.audio_id}.wav`;
    const audio = new Audio(audioUrl);

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

  const filteredWords = frequentVietnameseWords.filter((wordItem) =>
    wordItem.word.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-indigo-500 via-purple-400 to-pink-300 pt-24">
      <div className="max-w-6xl mx-auto px-6 py-10">
        <h1 className="text-4xl font-extrabold text-center text-white mb-12 drop-shadow-lg">
          The 2000 Most Frequent Vietnamese Words
        </h1>
        <div className="relative mb-8">
          <input
            type="text"
            placeholder="Search for a word..."
            className="w-full px-4 py-2 text-lg rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-purple-600 bg-white bg-opacity-80 text-indigo-900 placeholder-gray-400 transition-all"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-8">
          {filteredWords.map((wordItem: VietnameseWord, index: number) => {
            const { word, partOfSpeech, frequency } = wordItem;

            return (
              <div
                key={index}
                className="relative p-6 bg-white bg-opacity-80 rounded-xl shadow-md transition-transform transform hover:scale-105 hover:shadow-xl hover:bg-opacity-100"
              >
                <div
                  className="absolute top-0 left-0 h-2 rounded-t-xl bg-gradient-to-r from-green-400 via-blue-500 to-indigo-600 shadow-md"
                  style={{
                    width: `${Math.max(Math.min(100 - frequency * 10, 100), 0)}%`,
                    transition: 'width 0.3s ease-in-out',
                    boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.2)',
                  }}
                ></div>
                <h2 className="text-2xl font-bold text-indigo-800 mt-2">{word}</h2>
                <p className="italic text-gray-500 mb-4">{partOfSpeech}</p>
                <div className="flex justify-center space-x-6 mt-4">
                  <button
                    onClick={() => playSound(wordItem, 2)}
                    onMouseEnter={() => handleMouseEnter(index, 'northern')}
                    onMouseLeave={handleMouseLeave}
                    className="text-indigo-600 hover:text-indigo-800 focus:outline-none transition-colors transform hover:scale-110 duration-300"
                    aria-label="Northern Sound"
                  >
                    {hoveredIndex === index && hoveredType === 'northern' ? (
                      <span className="bg-indigo-600 text-white px-2 py-1 rounded-lg transition-opacity duration-300 ease-in-out">
                        NTH
                      </span>
                    ) : (
                      <i className="fa fa-volume-up text-xl"></i>
                    )}
                  </button>
                  <button
                    onClick={() => playSound(wordItem, 1)}
                    onMouseEnter={() => handleMouseEnter(index, 'southern')}
                    onMouseLeave={handleMouseLeave}
                    className="text-pink-600 hover:text-pink-800 focus:outline-none transition-colors transform hover:scale-110 duration-300"
                    aria-label="Southern Sound"
                  >
                    {hoveredIndex === index && hoveredType === 'southern' ? (
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
          Made with <span className="text-red-500 animate-pulse">❤️</span> by <strong className="text-indigo-800">{DEVELOPMENT_TEAM}</strong>
        </div>
        <div className="text-center text-gray-600 mt-2">
          <span className="font-semibold">Version {BUILD_VERSION}</span>
        </div>
      </footer>
    </div>
  );
}

export default Home;  