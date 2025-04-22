// /Users/bytedance/Documents/毕设/graduate-design/traffic-recognition-v2/frontend/src/components/ImageUpload.tsx
import React, { useState, useCallback } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast"; // Assuming you'll add toast later

interface ImageUploadProps {
  onImageSelect: (file: File | null) => void;
  isUploading: boolean;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageSelect,
  isUploading,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const { toast } = useToast(); // Placeholder for toast notifications

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        if (file.type.startsWith("image/")) {
          setSelectedFile(file);
          onImageSelect(file);
          // Create image preview
          const reader = new FileReader();
          reader.onloadend = () => {
            setPreviewUrl(reader.result as string);
          };
          reader.readAsDataURL(file);
        } else {
          // Handle non-image file selection
          setSelectedFile(null);
          onImageSelect(null);
          setPreviewUrl(null);
          event.target.value = ""; // Reset input
          toast({
            title: "文件类型错误",
            description: "请选择一个图片文件 (jpg, png, webp, etc.)。",
            variant: "destructive",
          });
        }
      } else {
        setSelectedFile(null);
        onImageSelect(null);
        setPreviewUrl(null);
      }
    },
    [onImageSelect, toast]
  );

  const handleRemoveImage = useCallback(() => {
    setSelectedFile(null);
    onImageSelect(null);
    setPreviewUrl(null);
    // Reset the file input visually if possible (might need a ref)
    const input = document.getElementById(
      "image-upload-input"
    ) as HTMLInputElement;
    if (input) input.value = "";
  }, [onImageSelect]);

  return (
    <div className="space-y-4">
      <Label htmlFor="image-upload-input">选择图片文件</Label>
      <Input
        id="image-upload-input"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        disabled={isUploading}
        className="cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
      />
      {previewUrl && (
        <div className="mt-4 relative group">
          <img
            src={previewUrl}
            alt="图片预览"
            className="max-w-full h-auto rounded-md border"
          />
          <Button
            variant="destructive"
            size="icon"
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={handleRemoveImage}
            disabled={isUploading}
            aria-label="移除图片"
          >
            {/* You might need to install lucide-react if not already done */}
            {/* <X className="h-4 w-4" /> */}
            <span>&times;</span> {/* Simple X as fallback */}
          </Button>
        </div>
      )}
      {!selectedFile && (
        <p className="text-sm text-muted-foreground">
          请选择一张图片进行识别。
        </p>
      )}
    </div>
  );
};
