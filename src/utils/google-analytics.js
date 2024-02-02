import ReactGA from "react-ga4";

export const initializeGA = () => {
  ReactGA.initialize(import.meta.env.VITE_APP_GA_TRACKING_ID);
};