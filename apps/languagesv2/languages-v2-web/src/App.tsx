import React, { useState } from 'react';
import './App.css';
import { FaCheckCircle, FaTimesCircle } from 'react-icons/fa';

const questions = [
  {
    questionText: 'What is the meaning of the Chinese word “爱” (ài)?',
    answerOptions: [
      { answerText: 'Love', isCorrect: true },
      { answerText: 'Water', isCorrect: false },
      { answerText: 'Fire', isCorrect: false },
      { answerText: 'Tree', isCorrect: false },
    ],
  },
  {
    questionText: 'What is the meaning of the Chinese word “水” (shuǐ)?',
    answerOptions: [
      { answerText: 'Water', isCorrect: true },
      { answerText: 'Mountain', isCorrect: false },
      { answerText: 'Earth', isCorrect: false },
      { answerText: 'Sky', isCorrect: false },
    ],
  },
  {
    questionText: 'What is the meaning of the Chinese word “火” (huǒ)?',
    answerOptions: [
      { answerText: 'Fire', isCorrect: true },
      { answerText: 'Flower', isCorrect: false },
      { answerText: 'Sun', isCorrect: false },
      { answerText: 'Moon', isCorrect: false },
    ],
  },
  {
    questionText: 'What is the meaning of the Chinese word “大” (dà)?',
    answerOptions: [
      { answerText: 'Big', isCorrect: true },
      { answerText: 'Small', isCorrect: false },
      { answerText: 'Happy', isCorrect: false },
      { answerText: 'Fast', isCorrect: false },
    ],
  },
  {
    questionText: 'What is the meaning of the Chinese word “小” (xiǎo)?',
    answerOptions: [
      { answerText: 'Small', isCorrect: true },
      { answerText: 'Tall', isCorrect: false },
      { answerText: 'Big', isCorrect: false },
      { answerText: 'Slow', isCorrect: false },
    ],
  },
  {
    questionText: 'What is the meaning of the Chinese word “男” (nán)?',
    answerOptions: [
      { answerText: 'Male', isCorrect: true },
      { answerText: 'Female', isCorrect: false },
      { answerText: 'Child', isCorrect: false },
      { answerText: 'Old', isCorrect: false },
    ],
  },
  {
    questionText: 'What is the meaning of the Chinese word “女” (nǔ)?',
    answerOptions: [
      { answerText: 'Female', isCorrect: true },
      { answerText: 'Male', isCorrect: false },
      { answerText: 'Young', isCorrect: false },
      { answerText: 'Tall', isCorrect: false },
    ],
  },
  {
    questionText: 'What is the meaning of the Chinese word “月” (yuè)?',
    answerOptions: [
      { answerText: 'Moon', isCorrect: true },
      { answerText: 'Star', isCorrect: false },
      { answerText: 'Sun', isCorrect: false },
      { answerText: 'Sky', isCorrect: false },
    ],
  },
  {
    questionText: 'What is the meaning of the Chinese word “日” (rì)?',
    answerOptions: [
      { answerText: 'Sun', isCorrect: true },
      { answerText: 'Moon', isCorrect: false },
      { answerText: 'Water', isCorrect: false },
      { answerText: 'Fire', isCorrect: false },
    ],
  },
  {
    questionText: 'What is the meaning of the Chinese word “学” (xué)?',
    answerOptions: [
      { answerText: 'Study', isCorrect: true },
      { answerText: 'Run', isCorrect: false },
      { answerText: 'Eat', isCorrect: false },
      { answerText: 'Play', isCorrect: false },
    ],
  },
];

function App() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [showScore, setShowScore] = useState(false);
  const [score, setScore] = useState(0);
  const [selectedAnswerIndex, setSelectedAnswerIndex] = useState<number | null>(null);
  const [isAnswerCorrect, setIsAnswerCorrect] = useState<boolean | null>(null);

  const handleAnswerOptionClick = (isCorrect: boolean, index: number) => {
    setSelectedAnswerIndex(index);
    setIsAnswerCorrect(isCorrect);
    if (isCorrect) {
      setScore(score + 1);
    }
    setTimeout(() => {
      const nextQuestion = currentQuestion + 1;
      if (nextQuestion < questions.length) {
        setCurrentQuestion(nextQuestion);
        setSelectedAnswerIndex(null);
        setIsAnswerCorrect(null);
      } else {
        setShowScore(true);
      }
    }, 1000);
  };

  return (
    <div className="App flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <header className="App-header w-full max-w-md p-8 bg-white shadow-md rounded-lg">
        <div className="quiz-section">
          {showScore ? (
            <div className='score-section text-2xl font-semibold text-center'>
              You scored {score} out of {questions.length}
            </div>
          ) : (
            <div>
              <div className='question-section mb-6'>
                <div className='question-count text-lg font-medium mb-4'>
                  <span>Question {currentQuestion + 1}</span>/{questions.length}
                </div>
                <div className='question-text text-xl font-semibold'>{questions[currentQuestion].questionText}</div>
              </div>
              <div className="w-full bg-gray-300 h-4 rounded mb-6">
                <div
                  className="bg-blue-500 h-4 rounded"
                  style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
                ></div>
              </div>
              <div className='answer-section grid grid-cols-2 gap-4'>
                {questions[currentQuestion].answerOptions.map((answerOption, index) => (
                  <button
                    key={index}
                    onClick={() => handleAnswerOptionClick(answerOption.isCorrect, index)}
                    className={`py-2 px-4 rounded transition duration-200 flex items-center justify-center space-x-2 ${
                      selectedAnswerIndex !== null
                        ? index === selectedAnswerIndex
                          ? answerOption.isCorrect
                            ? 'bg-green-500 text-white'
                            : 'bg-red-500 text-white'
                          : answerOption.isCorrect
                          ? 'bg-green-500 text-white'
                          : 'bg-blue-500 text-white'
                        : 'bg-blue-500 text-white hover:bg-blue-700'
                    }`}
                    disabled={selectedAnswerIndex !== null}
                  >
                    <span>{answerOption.answerText}</span>
                  </button>
                ))}
              </div>
              {selectedAnswerIndex !== null && (
                <div className="flex items-center justify-center mt-4">
                  {isAnswerCorrect ? (
                    <FaCheckCircle className="text-green-500 text-6xl" />
                  ) : (
                    <FaTimesCircle className="text-red-500 text-6xl" />
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;