import { useState, useEffect } from "react";
import axios from "axios";

export default function App() {
  const [maskImage, setMaskImage] = useState(null);
  const [selectedFile, setSelectedImage] = useState(null);
  const [originalImageUrl, setOriginalImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);


  
  // Preview original image when selected
  useEffect(() => {
    if (!selectedFile) {
      setOriginalImageUrl(null);
      setMaskImage(null);
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setOriginalImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  const handleUpload = async () => {
    if (!selectedFile) return alert("Please upload a file!");

    setLoading(true);
    setMaskImage(null);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const res = await axios.post("http://localhost:5001/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMaskImage(res.data.mask);
    } catch (error) {
      console.error(error);
      alert("Error uploading file");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center px-4 py-10 bg-gradient-to-br from-[#0f0f0f] to-[#1a0b2e]">
      <h1 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-[#6366F1] via-[#8B5CF6] to-[#EC4899] p-4 text-center mb-6 drop-shadow-lg">
        Satellite Image Explorer
      </h1>

      <p className="text-xl text-gray-300 max-w-2xl text-center mb-10">
        Upload a satellite image to see segmentation masks showing roads, buildings, vegetation, and more.
      </p>

      <div className="flex flex-col md:flex-row items-center gap-4 mb-10">
        <input
          id="imgFile"
          type="file"
          accept="image/*"
          onChange={(e) => setSelectedImage(e.target.files[0])}
          className="text-white rounded px-3 py-2 bg-gray-800 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-purple-600"
        />
        <button
          onClick={handleUpload}
          disabled={loading || !selectedFile}
          className="bg-gradient-to-r from-purple-600 to-fuchsia-600 text-white font-medium py-3 px-8 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Processing..." : "Upload and Predict"}
        </button>
      </div>

      {(originalImageUrl || maskImage) && (
        <div className="flex flex-col md:flex-row justify-center gap-12 w-full max-w-6xl">
          {originalImageUrl && (
            <div className="flex flex-col items-center">
              <h2 className="text-xl font-semibold mb-3 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-fuchsia-400">
                Original Image
              </h2>
              <img
                src={originalImageUrl}
                alt="Original Upload"
                className="rounded-xl shadow-lg max-w-xs md:max-w-md"
              />
            </div>
          )}

          {maskImage && (
            <div className="flex flex-col items-center">
              <h2 className="text-xl font-semibold mb-3 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-fuchsia-400">
                Predicted Mask
              </h2>
              <img
                src={maskImage}
                alt="Segmentation Mask"
                className="rounded-xl shadow-lg max-w-xs md:max-w-md"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
