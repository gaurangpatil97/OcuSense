'use client';

import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';

// Types
interface PredictionResponse {
  prediction: string;
  confidence: number;
  info: string;
  all_probs: { [disease: string]: number };
  model: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      handleFileSelection(droppedFile);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select an image file.');
      return;
    }
    
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const clearSelection = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Error during prediction:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred during prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to get color based on prediction
  const getBadgeColor = (prediction: string) => {
    return prediction.toLowerCase() === 'normal' 
      ? 'bg-[#22c55e] text-white' 
      : 'bg-[#ef4444] text-white';
  };

  return (
    <div className="flex-1 bg-[#0a0f1e] text-[#f1f5f9] font-sans selection:bg-[#3b82f6] selection:text-white flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0a0f1e]/80 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#3b82f6] to-[#1e3a8a] flex items-center justify-center shadow-[0_0_15px_rgba(59,130,246,0.5)]">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                <circle cx="12" cy="12" r="3"/>
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">OcuSense</h1>
              <p className="text-xs text-gray-400 font-medium tracking-wide uppercase">See beyond the image</p>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start h-full">
          
          {/* Left Column: Upload Area */}
          <div className="flex flex-col gap-6">
            <div className="bg-[#111827] rounded-2xl p-6 border border-gray-800 shadow-xl">
              <h2 className="text-xl font-semibold mb-4 text-white flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[#3b82f6]">
                  <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
                  <circle cx="9" cy="9" r="2"/>
                  <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
                </svg>
                Retinal Scan Upload
              </h2>
              
              {!previewUrl ? (
                <div 
                  className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${isDragging ? 'border-[#3b82f6] bg-[#3b82f6]/10' : 'border-gray-700 hover:border-gray-500 hover:bg-gray-800/50'}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    accept="image/*"
                    onChange={handleFileInputChange}
                  />
                  <div className="w-16 h-16 mb-4 rounded-full bg-gray-800 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-400">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" x2="12" y1="3" y2="15"/>
                    </svg>
                  </div>
                  <p className="text-lg font-medium text-white mb-1">Drag and drop your image</p>
                  <p className="text-sm text-gray-400">or click to browse files</p>
                  <p className="text-xs text-gray-500 mt-4">Supports JPG, PNG, JPEG</p>
                </div>
              ) : (
                <div className="flex flex-col gap-4">
                  <div className="relative rounded-xl overflow-hidden border border-gray-700 bg-black aspect-[4/3] flex items-center justify-center">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={previewUrl} alt="Retinal scan preview" className="max-w-full max-h-full object-contain" />
                    
                    {isLoading && (
                      <div className="absolute inset-0 bg-[#0a0f1e]/80 backdrop-blur-sm flex flex-col items-center justify-center z-10">
                        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-[#3b82f6] mb-4"></div>
                        <p className="text-[#3b82f6] font-medium animate-pulse">Analyzing scan...</p>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-3">
                    <button 
                      onClick={clearSelection}
                      disabled={isLoading}
                      className="flex-1 py-3 px-4 rounded-lg border border-gray-700 text-gray-300 font-medium hover:bg-gray-800 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Clear
                    </button>
                    <button 
                      onClick={handleSubmit}
                      disabled={isLoading || !file}
                      className="flex-[2] py-3 px-4 rounded-lg bg-[#3b82f6] text-white font-medium hover:bg-blue-600 shadow-[0_0_15px_rgba(59,130,246,0.3)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {isLoading ? (
                        <>
                          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Processing...
                        </>
                      ) : (
                        <>
                          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M2 12h4l2-9 5 18 2-9h5"/>
                          </svg>
                          Analyze Image
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}
              
              {error && (
                <div className="mt-4 p-4 rounded-lg bg-red-500/10 border border-red-500/50 text-red-400 flex items-start gap-3">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0 mt-0.5">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" x2="12" y1="8" y2="12"/>
                    <line x1="12" x2="12.01" y1="16" y2="16"/>
                  </svg>
                  <p className="text-sm">{error}</p>
                </div>
              )}
            </div>
            
            {/* Info Card */}
            <div className="bg-[#111827] rounded-2xl p-6 border border-gray-800 text-sm text-gray-400 shadow-lg mt-auto">
              <h3 className="text-white font-medium mb-2 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="M12 16v-4"/>
                  <path d="M12 8h.01"/>
                </svg>
                Instructions
              </h3>
              <p className="mb-2">Upload a clear, well-lit fundus image for best results. Blurry or poorly cropped images may reduce model accuracy.</p>
              <p>The system analyzes the image for common retinal conditions. Results are for informational purposes only and do not replace professional medical diagnosis.</p>
            </div>
          </div>

          {/* Right Column: Results */}
          <div className="flex flex-col h-full">
            <div className={`bg-[#111827] rounded-2xl p-6 border border-gray-800 shadow-xl h-full flex flex-col transition-all duration-500 ${result ? 'opacity-100 translate-y-0' : 'opacity-50'}`}>
              <h2 className="text-xl font-semibold mb-6 text-white flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[#3b82f6]">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <path d="M12 18v-6"/>
                  <path d="m9 15 3-3 3 3"/>
                </svg>
                Analysis Results
              </h2>

              {!result ? (
                <div className="flex-1 flex flex-col items-center justify-center text-center text-gray-500 min-h-[300px]">
                  <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="mb-4 opacity-20">
                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
                    <path d="M16 2v4h4"/>
                    <circle cx="14" cy="14" r="3"/>
                    <path d="m16 16 3.5 3.5"/>
                  </svg>
                  <p>Upload an image and run analysis<br/>to view results here.</p>
                </div>
              ) : (
                <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500 h-full">
                  {/* Primary Prediction */}
                  <div className="bg-[#0a0f1e] rounded-xl p-5 border border-gray-800">
                    <p className="text-gray-400 text-sm mb-1 uppercase tracking-wider font-semibold">Primary Finding</p>
                    <div className="flex items-center justify-between mb-4">
                      <span className={`text-2xl font-bold px-4 py-1.5 rounded-lg shadow-lg ${getBadgeColor(result.prediction)}`}>
                        {result.prediction}
                      </span>
                      <div className="text-right">
                        <span className="text-3xl font-light text-white">{result.confidence.toFixed(1)}</span>
                        <span className="text-gray-400 text-sm">%</span>
                      </div>
                    </div>
                    
                    {/* Confidence Bar */}
                    <div className="w-full bg-gray-800 rounded-full h-2.5 mb-1 overflow-hidden">
                      <div 
                        className={`h-2.5 rounded-full ${result.prediction.toLowerCase() === 'normal' ? 'bg-[#22c55e]' : 'bg-[#ef4444]'}`}
                        style={{ width: `${result.confidence}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-500 text-right">Confidence Score</p>
                  </div>

                  {/* Disease Info */}
                  <div>
                    <h3 className="text-white font-medium mb-2 border-b border-gray-800 pb-2">Clinical Context</h3>
                    <p className="text-gray-300 text-sm leading-relaxed bg-[#0a0f1e]/50 p-4 rounded-xl border border-gray-800/50">
                      {result.info}
                    </p>
                  </div>

                  {/* Probability Chart */}
                  <div className="flex-1 flex flex-col justify-end">
                    <h3 className="text-white font-medium mb-3 border-b border-gray-800 pb-2 flex justify-between items-end">
                      <span>Detailed Analysis</span>
                      {result.model && <span className="text-xs text-gray-500 font-normal">Model: {result.model}</span>}
                    </h3>
                    
                    <div className="flex flex-col gap-3 mt-2">
                      {Object.entries(result.all_probs)
                        .sort(([, a], [, b]) => b - a)
                        .map(([disease, prob]) => {
                          const percentage = prob.toFixed(1);
                          const isPrimary = disease === result.prediction;
                          const isNormal = disease.toLowerCase() === 'normal';
                          
                          return (
                            <div key={disease} className="flex flex-col gap-1">
                              <div className="flex justify-between text-xs">
                                <span className={`${isPrimary ? 'text-white font-medium' : 'text-gray-400'}`}>
                                  {disease}
                                </span>
                                <span className={isPrimary ? 'text-white' : 'text-gray-500'}>
                                  {percentage}%
                                </span>
                              </div>
                              <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                                <div 
                                  className={`h-1.5 rounded-full ${
                                    isPrimary 
                                      ? (isNormal ? 'bg-[#22c55e]' : 'bg-[#ef4444]') 
                                      : 'bg-[#3b82f6]/50'
                                  }`}
                                  style={{ width: `${prob}%` }}
                                ></div>
                              </div>
                            </div>
                          );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
          
        </div>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-gray-800 mt-auto bg-[#0a0f1e]/80">
        <div className="max-w-5xl mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between text-gray-500 text-sm">
          <p>© {new Date().getFullYear()} OcuSense AI. All rights reserved.</p>
          <p className="mt-2 md:mt-0">For research and clinical decision support only.</p>
        </div>
      </footer>
    </div>
  );
}
