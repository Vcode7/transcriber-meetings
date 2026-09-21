import React from 'react';
import TrainingLayout from '../features/training/components/TrainingLayout';

export default function Training() {
  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', minHeight: 0, overflow: 'hidden' }}>
      <TrainingLayout />
    </div>
  );
}
