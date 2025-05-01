import React from 'react';
import { Layout, Typography, Space, Button, Tooltip } from 'antd';
import { GithubOutlined, QuestionCircleOutlined } from '@ant-design/icons';

const { Header: AntHeader } = Layout;
const { Title } = Typography;

interface HeaderProps {
  onAboutClick?: () => void;
}

const Header: React.FC<HeaderProps> = ({ onAboutClick }) => {
  return (
    <AntHeader style={{ 
      background: '#fff', 
      padding: '0 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.09)'
    }}>
      <Title 
        level={3} 
        style={{ 
          margin: 0,
          color: '#1677ff'
        }}
      >
        深度学习交通标志识别系统
      </Title>
      
      <Space>
        <Tooltip title="查看帮助">
          <Button 
            type="text" 
            onClick={onAboutClick}
          >
            <QuestionCircleOutlined />
          </Button>
        </Tooltip>
        <Tooltip title="查看源码">
          <Button 
            type="text" 
            onClick={() => window.open('https://github.com/yourusername/traffic-recognition-v2', '_blank')}
          >
            <GithubOutlined />
          </Button>
        </Tooltip>
      </Space>
    </AntHeader>
  );
};

export default Header; 