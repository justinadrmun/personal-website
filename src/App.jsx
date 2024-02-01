import React, { useEffect } from 'react';
import ReactGA from 'react-ga';
import Animation from './components/Animation'
import LogoPanel from './components/LogoPanel'
import ActivityTimeline from './components/ActivityTimeline'

import './styles/app.css';

function App() {

  useEffect(() => {
    ReactGA.pageview(window.location.pathname + window.location.search);
  }, []);

  return (
        <div className="bg-neutral-100 outer-container">
              <div className="animation-container">
                <Animation/>
              </div>
              <div className="text-logo-container">
              <div className="text-container">
                justin <br/> munoz
              </div>
                <div className="logo-panel">
                  <LogoPanel />
                </div>
            </div>
        
            <div className="margin-top">
              <ActivityTimeline/>
            </div>
            
              <p className="footer-container">
                  &copy; {new Date().getFullYear()} Justin Munoz. All rights reserved.
              </p>
        </div>
  )
}

export default App