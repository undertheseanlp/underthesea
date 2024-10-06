import React, { useState, useRef } from 'react';
import { useParams } from 'react-router-dom';

type VideoData = {
  [key: string]: { title: string; url: string }[];
};

function Video() {
  const { language } = useParams<{ language: string }>();
  const [currentIndex, setCurrentIndex] = useState(0);
  const videoRef = useRef<HTMLIFrameElement>(null);

  const videoData: VideoData = {
    English: [
      {
        title: 'Gotye - Somebody That I Used To Know (feat. Kimbra)',
        url: 'https://www.youtube.com/embed/8UVNT4wvIGY?si=0ZDLyY8p4nvbSpaB&controls=0',
      },
      {
        title: 'MPOSSIBLE! [or NOT?] – Learn English Conversation in 4 Hours',
        url: 'https://www.youtube.com/embed/8FzY7cgKOmI?si=uADlutuyhPhXBkSc&controls=0',
      },
    ],
    Vietnamese: [
      {
        title: 'Vietnamese Phrases You Need at the Station',
        url: 'https://www.youtube.com/embed/0E50Kk0DNJk?si=U3dGZ77CUHb9VPoJ&controls=0',
      },
      {
        title: 'Learn Vietnamese in 2 Hours - Beginners Guide',
        url: 'https://www.youtube.com/embed/UvmzrMWD8_Y?si=e-DJH7gp00bS0yjZ&controls=0',
      },
    ],
    Chinese: [
      {
        title: '340 Chinese Words You\'ll Use Every Day - Basic Vocabulary #74',
        url: 'https://www.youtube.com/embed/40UHvFIJU6U?si=71k8LZKXJmu5oJAO&controls=0',
      },
      {
        title: 'Listening Practice - Naming Culture in China',
        url: 'https://www.youtube.com/embed/BCqjc388ExM?si=WDi8obbQM_fPEC22&controls=0',
      },
    ],
  };

  const handlePlay = () => {
    if (videoRef.current) {
      const iframeWindow = videoRef.current.contentWindow;
      iframeWindow?.postMessage('{"event":"command","func":"playVideo","args":""}', '*');
    }
  };

  const handlePause = () => {
    if (videoRef.current) {
      const iframeWindow = videoRef.current.contentWindow;
      iframeWindow?.postMessage('{"event":"command","func":"pauseVideo","args":""}', '*');
    }
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % videoData[language as string].length);
  };

  const currentVideo = videoData[language as string]?.[currentIndex];

  return (
    <div className="w-full p-16 space-y-24 bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Video Section */}
      {currentVideo && (
        <section className="space-y-6">
          <div className="bg-white shadow-xl rounded-lg p-6 transform hover:scale-105 transition-transform duration-300">
            <h2 className="text-3xl font-semibold mb-4 text-indigo-700">{currentVideo.title}</h2>
            <div className="relative pb-[56.25%]">
              <iframe
                ref={videoRef}
                title={currentVideo.title}
                src={`${currentVideo.url}&enablejsapi=1`}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="absolute top-0 left-0 w-full h-full rounded-lg shadow-md"
              ></iframe>
            </div>
            <div className="flex space-x-4 mt-6 justify-center">
              <button
                onClick={handlePlay}
                className="px-6 py-3 bg-indigo-500 text-white font-bold rounded-full shadow-md hover:bg-indigo-600 transform hover:scale-105 transition-transform duration-200"
              >
                ▶ Play
              </button>
              <button
                onClick={handlePause}
                className="px-6 py-3 bg-yellow-500 text-white font-bold rounded-full shadow-md hover:bg-yellow-600 transform hover:scale-105 transition-transform duration-200"
              >
                ⏸ Pause
              </button>
              <button
                onClick={handleNext}
                className="px-6 py-3 bg-green-500 text-white font-bold rounded-full shadow-md hover:bg-green-600 transform hover:scale-105 transition-transform duration-200"
              >
                ⏭ Next
              </button>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

export default Video;