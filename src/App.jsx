import React from 'react';
import Animation from './components/Animation.jsx'
import LogoPanel from './components/LogoPanel.jsx'
import ActivityTimeline from './components/ActivityTimeline.jsx'
import GoogleAnalytics from "./components/GoogleAnalytics.jsx";

import './styles/app.css';

function App() {
  return (
        <div className="bg-neutral-100 outer-container">
              <GoogleAnalytics/>
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