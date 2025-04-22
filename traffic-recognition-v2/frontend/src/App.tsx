import { useState, useCallback } from "react";
import { ImageUpload } from "@/components/ImageUpload";
import { ModelSelector } from "@/components/ModelSelector";
import { ResultDisplay } from "@/components/ResultDisplay";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { Toaster } from "@/components/ui/toaster"; // Import Toaster
import { uploadImageForRecognition } from "@/services/api";
import { ModelType, RecognitionResult } from "@/types";
import "./App.css"; // Keep existing App.css or replace with global styles import if needed
import "./index.css"; // Ensure Tailwind styles are loaded

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelType>("yolo"); // Default model
  const [results, setResults] = useState<RecognitionResult[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null); // For displaying the uploaded image with results
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const handleImageSelect = useCallback((file: File | null) => {
    setSelectedFile(file);
    setResults([]); // Clear previous results
    setError(null); // Clear previous errors
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setImageUrl(null);
    }
  }, []);

  const handleModelChange = useCallback((model: ModelType) => {
    setSelectedModel(model);
  }, []);

  const handleUploadAndRecognize = useCallback(async () => {
    if (!selectedFile) {
      toast({
        title: "未选择文件",
        description: "请先选择一张图片文件。",
        variant: "warning",
      });
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults([]); // Clear previous results before new request

    try {
      const response = await uploadImageForRecognition(
        selectedFile,
        selectedModel
      );
      if (response.code === 200 || response.code === 0) {
        // Assuming 0 or 200 means success
        setResults(response.data || []);
        if (!response.data || response.data.length === 0) {
          toast({
            title: "识别完成",
            description: "未检测到任何目标。",
          });
        }
      } else {
        setError(response.message || "识别失败，请稍后重试。");
        toast({
          title: "识别出错",
          description: response.message || "未知错误",
          variant: "destructive",
        });
      }
    } catch (err: any) {
      const message =
        err.message || "发生意外错误，请检查网络连接或联系管理员。";
      setError(message);
      toast({
        title: "请求失败",
        description: message,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile, selectedModel, toast]);

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold">交通标志识别系统</h1>
      </header>
      <main className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left Column: Upload and Controls */}
        <div className="space-y-6">
          <ImageUpload
            onImageSelect={handleImageSelect}
            isUploading={isLoading}
          />
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
            disabled={isLoading}
          />
          <Button
            onClick={handleUploadAndRecognize}
            disabled={isLoading || !selectedFile}
            className="w-full"
          >
            {isLoading ? <LoadingSpinner size="sm" className="mr-2" /> : null}
            {isLoading ? "正在识别..." : "上传并识别"}
          </Button>
          {error && <p className="text-red-600 text-sm">错误: {error}</p>}
        </div>

        {/* Right Column: Results */}
        <div className="space-y-6">
          {isLoading && !results.length ? (
            <div className="flex justify-center items-center h-64 border rounded-md">
              <LoadingSpinner />
            </div>
          ) : (
            <ResultDisplay imageUrl={imageUrl} results={results} />
          )}
        </div>
      </main>
      <Toaster /> {/* Add Toaster component here */}
    </div>
  );
}

export default App;
