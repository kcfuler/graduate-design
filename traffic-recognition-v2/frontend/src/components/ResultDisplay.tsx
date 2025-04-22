// /Users/bytedance/Documents/毕设/graduate-design/traffic-recognition-v2/frontend/src/components/ResultDisplay.tsx
import React, { useRef, useEffect, useState } from "react";
import { RecognitionResult } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area"; // Assuming you'll add this later
import { Badge } from "@/components/ui/badge"; // Assuming you'll add this later

interface ResultDisplayProps {
  imageUrl: string | null;
  results: RecognitionResult[];
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({
  imageUrl,
  results,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageSize, setImageSize] = useState<{
    width: number;
    height: number;
  } | null>(null);

  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Set canvas size to image size initially
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight });

      // Draw the image
      ctx.drawImage(img, 0, 0);

      // Draw bounding boxes
      results.forEach((result) => {
        const [xmin, ymin, xmax, ymax] = result.box;
        const label = `${result.label} (${(result.confidence * 100).toFixed(
          1
        )}%)`;

        // Style the box and text
        ctx.strokeStyle = "red"; // Example color
        ctx.lineWidth = 2;
        ctx.fillStyle = "red";
        ctx.font = "14px Arial";

        // Draw the rectangle
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        // Draw the label background
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(xmin, ymin - 18, textWidth + 4, 18);

        // Draw the label text
        ctx.fillStyle = "white";
        ctx.fillText(label, xmin + 2, ymin - 4);
      });
    };
    img.onerror = () => {
      console.error("Failed to load image for canvas drawing.");
      setImageSize(null);
      // Optionally clear canvas or show an error message
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    };
    img.src = imageUrl;
  }, [imageUrl, results]);

  // Adjust canvas display size based on container, maintaining aspect ratio
  const canvasStyle: React.CSSProperties = imageSize
    ? {
        maxWidth: "100%",
        height: "auto",
        aspectRatio: `${imageSize.width} / ${imageSize.height}`,
      }
    : { display: "none" };

  return (
    <Card>
      <CardHeader>
        <CardTitle>识别结果</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {imageUrl ? (
          <div className="relative border rounded-md overflow-hidden">
            <canvas ref={canvasRef} style={canvasStyle} />
            {!imageSize && (
              <p className="p-4 text-center text-muted-foreground">
                正在加载图片...
              </p>
            )}
          </div>
        ) : (
          <p className="text-muted-foreground">请先上传一张图片。</p>
        )}

        {results.length > 0 && (
          <div>
            <h4 className="font-semibold mb-2">检测到的目标:</h4>
            <ScrollArea className="h-[150px] w-full rounded-md border p-2">
              <ul className="space-y-1">
                {results.map((result, index) => (
                  <li
                    key={index}
                    className="text-sm flex justify-between items-center"
                  >
                    <span>{result.label}</span>
                    <Badge variant="secondary">
                      {(result.confidence * 100).toFixed(1)}%
                    </Badge>
                  </li>
                ))}
              </ul>
            </ScrollArea>
          </div>
        )}
        {imageUrl && results.length === 0 && (
          <p className="text-muted-foreground">未检测到任何目标。</p>
        )}
      </CardContent>
    </Card>
  );
};
