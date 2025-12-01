import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);
    
    if (selectedFile) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = async () => {
    if (!file) return alert("Please select an image.");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error connecting to server. Make sure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setPreview(null);
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
      padding: "20px"
    }}>
      <div style={{
        background: "white",
        borderRadius: "20px",
        boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
        padding: "40px",
        maxWidth: "600px",
        width: "100%"
      }}>
        <div style={{ textAlign: "center", marginBottom: "30px" }}>
          <h1 style={{
            fontSize: "2.5rem",
            fontWeight: "700",
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            marginBottom: "10px"
          }}>
            🔍 Deepfake Detection
          </h1>
          <p style={{ color: "#666", fontSize: "1rem" }}>
            Upload an image to detect if it's real or fake
          </p>
        </div>

        <div style={{
          border: "3px dashed #667eea",
          borderRadius: "15px",
          padding: "40px 20px",
          textAlign: "center",
          background: "#f8f9ff",
          marginBottom: "20px",
          transition: "all 0.3s ease"
        }}>
          {preview ? (
            <div>
              <img 
                src={preview} 
                alt="Preview" 
                style={{
                  maxWidth: "100%",
                  maxHeight: "300px",
                  borderRadius: "10px",
                  marginBottom: "15px",
                  boxShadow: "0 4px 15px rgba(0,0,0,0.1)"
                }}
              />
              <p style={{ color: "#667eea", fontWeight: "600", marginBottom: "10px" }}>
                📄 {file.name}
              </p>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: "4rem", marginBottom: "10px" }}>📁</div>
              <p style={{ color: "#666", fontSize: "1.1rem", marginBottom: "15px" }}>
                Drop your image here or click to browse
              </p>
            </div>
          )}
          
          <input 
            type="file" 
            onChange={handleFileChange} 
            accept="image/*"
            id="fileInput"
            style={{ display: "none" }}
          />
          <label 
            htmlFor="fileInput"
            style={{
              display: "inline-block",
              padding: "12px 30px",
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              color: "white",
              borderRadius: "25px",
              cursor: "pointer",
              fontWeight: "600",
              transition: "transform 0.2s ease",
              border: "none"
            }}
            onMouseEnter={(e) => e.target.style.transform = "scale(1.05)"}
            onMouseLeave={(e) => e.target.style.transform = "scale(1)"}
          >
            Choose Image
          </label>
        </div>

        <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
          <button 
            onClick={handleSubmit}
            disabled={!file || loading}
            style={{
              flex: 1,
              padding: "15px",
              fontSize: "1.1rem",
              fontWeight: "600",
              border: "none",
              borderRadius: "10px",
              background: loading ? "#ccc" : "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              color: "white",
              cursor: loading || !file ? "not-allowed" : "pointer",
              transition: "all 0.3s ease",
              opacity: loading || !file ? 0.6 : 1
            }}
            onMouseEnter={(e) => {
              if (!loading && file) e.target.style.transform = "translateY(-2px)";
            }}
            onMouseLeave={(e) => {
              if (!loading && file) e.target.style.transform = "translateY(0)";
            }}
          >
            {loading ? "🔄 Analyzing..." : "🚀 Analyze Image"}
          </button>
          
          {(file || result) && (
            <button 
              onClick={handleReset}
              style={{
                padding: "15px 25px",
                fontSize: "1.1rem",
                fontWeight: "600",
                border: "2px solid #667eea",
                borderRadius: "10px",
                background: "white",
                color: "#667eea",
                cursor: "pointer",
                transition: "all 0.3s ease"
              }}
              onMouseEnter={(e) => {
                e.target.style.background = "#667eea";
                e.target.style.color = "white";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "white";
                e.target.style.color = "#667eea";
              }}
            >
              Reset
            </button>
          )}
        </div>

        {result && (
          <div style={{
            marginTop: "30px",
            padding: "25px",
            borderRadius: "15px",
            background: result.prediction === "Real" ? "#d4edda" : "#f8d7da",
            border: `3px solid ${result.prediction === "Real" ? "#28a745" : "#dc3545"}`,
            animation: "fadeIn 0.5s ease"
          }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: "3rem", marginBottom: "10px" }}>
                {result.prediction === "Real" ? "✅" : "⚠️"}
              </div>
              <h2 style={{
                fontSize: "2rem",
                color: result.prediction === "Real" ? "#28a745" : "#dc3545",
                marginBottom: "10px",
                fontWeight: "700"
              }}>
                {result.prediction}
              </h2>
              <div style={{
                fontSize: "1.5rem",
                fontWeight: "600",
                color: "#333",
                marginBottom: "15px"
              }}>
                Confidence: {result.confidence}
              </div>
              
              {result.probabilities && (
                <div style={{
                  marginTop: "20px",
                  textAlign: "left",
                  background: "white",
                  padding: "15px",
                  borderRadius: "10px"
                }}>
                  <h4 style={{ marginBottom: "10px", color: "#666" }}>Detailed Probabilities:</h4>
                  {Object.entries(result.probabilities).map(([key, value]) => (
                    <div key={key} style={{ marginBottom: "8px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                        <span style={{ fontWeight: "600" }}>{key}:</span>
                        <span style={{ color: "#667eea" }}>{value}</span>
                      </div>
                      <div style={{
                        height: "8px",
                        background: "#e0e0e0",
                        borderRadius: "4px",
                        overflow: "hidden"
                      }}>
                        <div style={{
                          height: "100%",
                          width: value,
                          background: "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
                          transition: "width 0.5s ease"
                        }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      <style>
        {`
          @keyframes fadeIn {
            from {
              opacity: 0;
              transform: translateY(20px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>
    </div>
  );
}

export default App;