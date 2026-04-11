import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import '../App.css';
import { FaCheckCircle, FaTimesCircle } from 'react-icons/fa';
import questions from './quiz/QuizData';

type LanguageType = 'English' | 'Vietnamese' | 'Chinese'; // Add all the possible languages here

function Quiz() {
  const { search } = useLocation();
  const params = new URLSearchParams(search);
  const language = (params.get('language') || 'English') as LanguageType;
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [showScore, setShowScore] = useState(false);
  const [score, setScore] = useState(0);
  const [selectedAnswerIndex, setSelectedAnswerIndex] = useState<number | null>(null);
  const [isAnswerCorrect, setIsAnswerCorrect] = useState<boolean | null>(null);

  useEffect(() => {
    if (language && questions[language]) {
      setCurrentQuestion(0);
      setShowScore(false);
      setScore(0);
      setSelectedAnswerIndex(null);
      setIsAnswerCorrect(null);
    }
  }, [language]);

  const handleAnswerOptionClick = (isCorrect: boolean, index: number) => {
    setSelectedAnswerIndex(index);
    setIsAnswerCorrect(isCorrect);
    if (isCorrect) {
      setScore(score + 1);
    }
    setTimeout(() => {
      const nextQuestion = currentQuestion + 1;
      if (nextQuestion < questions[language].length) {
        setCurrentQuestion(nextQuestion);
        setSelectedAnswerIndex(null);
        setIsAnswerCorrect(null);
      } else {
        setShowScore(true);
      }
    }, 1000);
  };

  return (
    <div className="quiz-section mt-24">
      {showScore ? (
        <div className='score-section text-2xl font-semibold text-center'>
          You scored {score} out of {questions[language].length}
        </div>
      ) : (
        <div>
          <div className='question-section mb-6'>
            <div className='question-count text-lg font-medium mb-4'>
              <span>Question {currentQuestion + 1}</span>/{questions[language].length}
            </div>
            <div className='question-text text-xl font-semibold'>{questions[language][currentQuestion].questionText}</div>
          </div>
          <div className="w-full bg-gray-300 h-4 rounded mb-6">
            <div
              className="bg-blue-500 h-4 rounded"
              style={{ width: `${((currentQuestion + 1) / questions[language].length) * 100}%` }}
            ></div>
          </div>
          <div className='answer-section grid grid-cols-2 gap-4'>
            {questions[language][currentQuestion].answerOptions.map((answerOption, index) => (
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
  );
}

export default Quiz;