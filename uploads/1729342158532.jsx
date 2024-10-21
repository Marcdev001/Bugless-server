import React, { useState, useRef, useCallback, useEffect } from "react";
import axios from "axios";
import { Cropper } from "react-cropper";
import "cropperjs/dist/cropper.css";
import "../GetStarted.css";
import BackToTop from '../Components/BackToTop';

const backgroundDesigns = [
  { id: "design3", name: "Not Available", url: "https://example.com/background-designs/design3.jpg" },
];

const API_BASE_URL = "https://no-bg.onrender.com";

function GetStarted() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [bgColor, setBgColor] = useState("");
  const [backgroundDesign, setBackgroundDesign] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isCropping, setIsCropping] = useState(false);
  const cropperRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 500, height: 500 });
  const mediaContainerRef = useRef(null);
  const [isBgRemoved, setIsBgRemoved] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const updateDimensions = useCallback(() => {
    const width = window.innerWidth;
    if (width < 768) {
      setDimensions({ width: 300, height: 300 });
    } else if (width < 1024) {
      setDimensions({ width: 400, height: 400 });
    } else {
      setDimensions({ width: 500, height: 500 });
    }
  }, []);

  useEffect(() => {
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, [updateDimensions]);

  const handleFileChange = useCallback((e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(URL.createObjectURL(selectedFile));
    setError(null);
    setIsBgRemoved(false);
    setBgColor("");
    setBackgroundDesign("");
  }, []);

  const processImage = useCallback(async (endpoint, formData) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      });
      const url = URL.createObjectURL(response.data);
      setResult(url);
      if (endpoint === '/remove-bg') {
        setIsBgRemoved(true);
      }
    } catch (error) {
      console.error(`Error processing image:`, error);
      setError(`An error occurred while processing the image. Please try again.`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleRemoveBackground = useCallback(async (event) => {
    event.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append("image", file);
    if (bgColor && isBgRemoved) {
      formData.append("bgColor", bgColor);
    }
    if (backgroundDesign && isBgRemoved) {
      formData.append("backgroundDesign", backgroundDesign);
    }
    await processImage('/remove-bg', formData);
  }, [file, bgColor, backgroundDesign, isBgRemoved, processImage]);

  const handleApplyBackground = useCallback(() => {
    if (!isBgRemoved) {
      setError('Remove the original background before applying background color or design.');
      return;
    }
    handleRemoveBackground(new Event('submit'));
  }, [isBgRemoved, handleRemoveBackground]);

  const handleCrop = useCallback(async () => {
    if (cropperRef.current && cropperRef.current.cropper) {
      const croppedCanvas = cropperRef.current.cropper.getCroppedCanvas();
      const formData = new FormData();
      croppedCanvas.toBlob(async (blob) => {
        formData.append("file", blob, "cropped-image.png");
        const cropData = cropperRef.current.cropper.getData();
        formData.append("width", Math.round(cropData.width));
        formData.append("height", Math.round(cropData.height));
        formData.append("x", Math.round(cropData.x));
        formData.append("y", Math.round(cropData.y));
        await processImage('/crop-image', formData);
        setIsCropping(false);
      });
    }
  }, [processImage]);

  const handleDownload = useCallback(() => {
    if (result) {
      const link = document.createElement("a");
      link.href = result;
      link.download = "no-bg.png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }, [result]);

  const renderMedia = useCallback(() => {
    if (!file) return null;
    const url = result || URL.createObjectURL(file);
    const style = {
      width: `${dimensions.width}px`,
      height: `${dimensions.height}px`,
      objectFit: "contain",
      backgroundColor: bgColor || undefined,
    };
    return <img src={url} alt="Preview" style={style} />;
  }, [file, result, dimensions, bgColor]);

  return (
    <div className={`App ${isVisible ? "fade-in" : ""}`}>
      <h1>Advanced Image Editing App</h1>
      <p className="getstarted-guide"><div><i class="fa-brands fa-twitter fa-beat icon1"></i></div>Upload your image to start editing and scroll down to see your uploaded image</p>
      <form onSubmit={handleRemoveBackground}>
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <input
          className='Enter-color'
          type="text"
          placeholder="Enter background color (optional)"
          value={bgColor}
          onChange={(e) => setBgColor(e.target.value)}
        />
        <select
          value={backgroundDesign}
          onChange={(e) => setBackgroundDesign(e.target.value)}
        >
          <option value="">Select background design (optional)</option>
          {backgroundDesigns.map((design) => (
            <option key={design.id} value={design.url}>
              {design.name}
            </option>
          ))}
        </select>
        <button type="submit" disabled={!file || isLoading}>Remove Background</button>
        <button type="button" onClick={handleApplyBackground} disabled={!isBgRemoved || isLoading}>
          Apply Color
        </button>
        <button type="button" onClick={() => setIsCropping(true)} disabled={!file || isLoading}>
          Crop
        </button>
        <button type="button" onClick={handleDownload} disabled={!result || isLoading}>
          Download
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {isLoading && <p><b>Loading...</b></p>}
      <div className="result-section" ref={mediaContainerRef}>
        {!isCropping && renderMedia()}
        {isCropping && (
          <div style={{ width: `${dimensions.width}px`, height: `${dimensions.height}px` , marginBottom: "4rem"}}>
            <Cropper
              src={result || URL.createObjectURL(file)}
              style={{ height: "100%", width: "100%" }}
              initialAspectRatio={1}
              guides={false}
              ref={cropperRef}
            />
            <button className="Apply-Crop" onClick={handleCrop}>Apply Crop</button>
            <button className="Apply-cancel" onClick={() => setIsCropping(false)}>Cancel</button>
          </div>
        )}
      </div>
      <BackToTop />
    </div>
  );
}

export default GetStarted;