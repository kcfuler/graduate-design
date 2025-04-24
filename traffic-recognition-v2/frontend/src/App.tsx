import React from 'react';
import Header from './components/Header';
import ModelSelector from './components/ModelSelector';
import MediaUploader from './components/MediaUploader';
import ResultDisplay from './components/ResultDisplay';
import MetricsDisplay from './components/MetricsDisplay';
import TrainingPanel from './components/TrainingPanel';
// import './App.css'; // 如果 index.css 已导入 Tailwind，这个可能不再需要

function App() {
  return (
    <div className="container mx-auto p-4">
      <Header />
      <main className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <section>
          <ModelSelector />
          <MediaUploader />
          <TrainingPanel />
        </section>
        <section>
          <ResultDisplay />
          <MetricsDisplay />
        </section>
      </main>
    </div>
  );
}

export default App;
