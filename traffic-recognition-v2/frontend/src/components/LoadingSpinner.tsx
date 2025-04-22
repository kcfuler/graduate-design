// /Users/bytedance/Documents/毕设/graduate-design/traffic-recognition-v2/frontend/src/components/LoadingSpinner.tsx
import React from "react";

interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg"; // 控制大小
  className?: string; // 允许传入额外样式
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = "md",
  className = "",
}) => {
  const sizeClasses = {
    sm: "h-4 w-4 border-2",
    md: "h-8 w-8 border-4",
    lg: "h-12 w-12 border-4",
  };

  return (
    <div className="flex justify-center items-center">
      <div
        className={`animate-spin rounded-full border-primary border-t-transparent ${sizeClasses[size]} ${className}`}
        role="status"
        aria-live="polite"
        aria-label="加载中"
      >
        <span className="sr-only">加载中...</span>
      </div>
    </div>
  );
};
