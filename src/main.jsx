import React from 'react'
import ReactDOM from 'react-dom/client'
import ReactGA from 'react-ga';
import App from './App.jsx'
import './styles/tailwind.css'

ReactGA.initialize(import.meta.env.VITE_APP_GA_TRACKING_ID);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
