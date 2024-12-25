import React, { useState, useEffect } from 'react';
import Marquee from 'react-fast-marquee';
import '../styles/animation.css';

const Animation = () => {
  const textArrays = [
    ['subject matter expert','data science','cloud automation'],
    ['API development','data visualisation','data analytics'],
    ['machine learning','graph neural networks','researcher'],
    ['R','microsoft azure','SQL', 'shinyapps','streamlit'],
    ['python','fintech','natural language processing','PowerBI']
  ];

  // create random number between min and max
  const getRandomDuration = () => {
    let min, max;
  
    if (window.innerWidth < 768) {min = 130; max = 170;} 
    if ((window.innerWidth >= 768) & (window.innerWidth < 1024)) {min = 150; max = 190;}
    if (window.innerWidth >= 1024) {min = 160; max = 200;}
  
    return Math.random() * (max - min) + min;
  };

  const [durations, setDurations] = useState(textArrays.map(() => getRandomDuration()));
  const [isResizing, setIsResizing] = useState(false);

  useEffect(() => {
    let resizeTimer;
    const handleResize = () => {
      if (window.innerWidth > 450) {
        setIsResizing(true);
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          setIsResizing(false);
          setDurations(durations.map(() => getRandomDuration()));
        }, 500);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [durations]);

  return (
    <div className="relative">
      {isResizing ? null : textArrays.map((textArray, i) => (
        <Marquee className={`marquee-${i+1}`} speed={durations[i]} pauseOnHover={false} pause={isResizing} key={i}>
          {textArray.sort(() => Math.random() - 0.5).map((text, index) => (
            <div className={`animated-string${i+1}`} key={index}>
              <span>{text}</span>
            </div>
          ))}
        </Marquee>
      ))}
    </div>
  );
}

export default Animation;