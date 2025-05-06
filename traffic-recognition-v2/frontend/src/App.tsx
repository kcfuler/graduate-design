import React from 'react';
import Layout from 'antd/es/layout';
const { Content } = Layout;
import Header from './components/Header';
import InferenceLayout from './components/InferenceLayout';

const App: React.FC = () => {
  return (
    <Layout className="min-h-screen">
      <Header />
      <Content>
        <InferenceLayout />
      </Content>
    </Layout>
  );
};

export default App; 