import React, { useEffect } from 'react';
import PropTypes from 'prop-types';
import { styled } from '@mui/material/styles';
import Stack from '@mui/material/Stack';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import StepConnector, { stepConnectorClasses } from '@mui/material/StepConnector';
import { Typography } from '@mui/material';

import '../styles/activity-timeline.css';

const QontoConnector = styled(StepConnector)(({ theme }) => ({
  [`&.${stepConnectorClasses.alternativeLabel}`]: {
    top: 10,
    left: 'calc(-50% + 16px)',
    right: 'calc(50% + 16px)',
  },
  [`&.${stepConnectorClasses.active}`]: {
    [`& .${stepConnectorClasses.line}`]: {
      borderColor: '#eaeaf0',
    },
  },
  [`& .${stepConnectorClasses.line}`]: {
    borderColor: theme.palette.mode === 'dark' ? theme.palette.grey[800] : '#eaeaf0',
    borderTopWidth: 3,
    borderRadius: 1,
  },
}));

const QontoStepIconRoot = styled('div')(({ theme, ownerState }) => ({
  color: theme.palette.mode === 'dark' ? theme.palette.grey[700] : '#235977',
  display: 'flex',
  height: 22,
  alignItems: 'center',
  ...(ownerState.active && {
    color: '#235977',
  }),
  '& .QontoStepIcon-circle': {
    width: 8,
    height: 8,
    borderRadius: '50%',
    backgroundColor: 'transparent',
    border: '2px solid currentColor',
  },
}));

function QontoStepIcon(props) {
  const { active, className, date } = props;

  return (
    <div className='stepicon-container'>
      <Typography variant="body1" color="#235977" className='stepicon-text' style={{fontWeight: 600, fontSize: '0.9rem'
      }}>{date}</Typography>
      <QontoStepIconRoot ownerState={{ active }} className={className}>
        {<div className="QontoStepIcon-circle" />}
      </QontoStepIconRoot>
    </div>
  );
}

QontoStepIcon.propTypes = {
  /**
   * Whether this step is active.
   * @default false
   */
  active: PropTypes.bool,
  className: PropTypes.string,
  date: PropTypes.string,
};

const steps = [
  { 
    date: "Dec '23", 
    activity: [
      { text: 'Presented at\n', url: null },
      { text: 'CUDAN conference', url: 'https://youtu.be/nP0dsi6wgPA?feature=shared&t=15710' }
    ]
  },
  { 
    date: "Jan '24", 
    activity: [
      { text: 'Submitted a ML-related\n', url: null },
      { text: 'manuscript', url: 'http://dx.doi.org/10.2139/ssrn.4698441' }
    ]
  },
  { 
    date: "Feb '24", 
    activity: [
      { text: 'Created this ', url: null },
      { text: 'website', url: 'https://github.com/justinadrmun/personal-website' }
    ]
  },
];

export default function ActivityTimeline() {
  useEffect(() => {
    const links = document.querySelectorAll('.activity-link');
    links.forEach(link => {
      link.addEventListener('click', () => {
        link.blur();
      });
    });
  }, []);
  
  return (
    <Stack sx={{ width: '100%' }} spacing={4}>
      <h2 className="timeline-title">Recent activity</h2>
      <Stepper alternativeLabel activeStep={2} connector={<QontoConnector />}>
      {steps.map((step, stepIndex) => (
        <Step key={stepIndex}>
          <StepLabel StepIconComponent={(props) => <QontoStepIcon {...props} date={step.date} />}>
            <Typography variant="body2" className="activity-text">
              {step.activity.map((item, linkIndex) => (
                <React.Fragment key={linkIndex}>
                  {item.url ? (
                    <a 
                      href={item.url} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className={`activity-link activity-link-${stepIndex}-${linkIndex}`}
                    >
                      {item.text}
                    </a>
                  ) : (
                    item.text
                  )}
                  {item.text.endsWith('\n') && <br />}
                </React.Fragment>
              ))}
            </Typography>
          </StepLabel>
        </Step>
      ))}
      </Stepper>
    </Stack>
  );
}