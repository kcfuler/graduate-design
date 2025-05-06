import React from 'react';
import { Layout } from 'antd';
import Header from './components/Header';
import InferenceLayout from './components/InferenceLayout';

const App: React.FC = () => {
  return (
    <Layout className="min-h-screen">
      <Header />
      <Layout.Content>
        <InferenceLayout />
      </Layout.Content>
    </Layout>
  );
};

export default App; 