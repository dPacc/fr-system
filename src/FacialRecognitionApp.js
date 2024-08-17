import React, { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";
import "@tensorflow/tfjs-backend-webgl";
import "./App.css";

const FacialRecognitionApp = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [capturedImages, setCapturedImages] = useState([]);
  const [recognizedName, setRecognizedName] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [faceMatcher, setFaceMatcher] = useState(null);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (isModelLoaded) {
      loadStoredDescriptors();
    }
  }, [isModelLoaded]);

  const loadModels = async () => {
    const MODEL_URL =
      "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights";

    console.log("Attempting to load models from:", MODEL_URL);

    try {
      console.log("--- Loading SSD Mobilenet model ---");
      await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
      console.log("SSD Mobilenet model loaded successfully");

      console.log("--- Loading Face Landmark model ---");
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      console.log("Face Landmark model loaded successfully");

      console.log("--- Loading Face Recognition model ---");
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
      console.log("Face Recognition model loaded successfully");

      setIsModelLoaded(true);
      console.log("=== Face recognition models loaded ===");
    } catch (error) {
      console.error("Error loading models:", error);
      console.error("Error details:", error.message);
      if (error.stack) {
        console.error("Error stack:", error.stack);
      }

      // Log the current state of each model
      console.log(
        "SSD Mobilenet model loaded:",
        faceapi.nets.ssdMobilenetv1.isLoaded
      );
      console.log(
        "Face Landmark model loaded:",
        faceapi.nets.faceLandmark68Net.isLoaded
      );
      console.log(
        "Face Recognition model loaded:",
        faceapi.nets.faceRecognitionNet.isLoaded
      );
    }
  };

  const loadStoredDescriptors = () => {
    const storedDescriptors = JSON.parse(
      localStorage.getItem("faceDescriptors") || "[]"
    );
    if (storedDescriptors.length > 0) {
      const labeledFaceDescriptors = storedDescriptors.map(
        (descriptor) =>
          new faceapi.LabeledFaceDescriptors(
            descriptor.label,
            descriptor.descriptors.map((d) => new Float32Array(d))
          )
      );
      setFaceMatcher(new faceapi.FaceMatcher(labeledFaceDescriptors));
    }
  };

  const captureImage = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setCapturedImages((prev) => [...prev, imageSrc]);
    }
  };

  const trainModel = async () => {
    if (capturedImages.length < 5) {
      alert("Please capture at least 5 images before training");
      return;
    }

    const personName = prompt("Please enter the person's name:");
    if (!personName) return;

    setIsTraining(true);
    const descriptions = [];

    for (let i = 0; i < capturedImages.length; i++) {
      const img = await faceapi.fetchImage(capturedImages[i]);
      const detection = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        descriptions.push(detection.descriptor);
      }
    }

    if (descriptions.length > 0) {
      const newFaceDescriptor = new faceapi.LabeledFaceDescriptors(
        personName,
        descriptions
      );

      // Get existing descriptors
      const existingDescriptors = JSON.parse(
        localStorage.getItem("faceDescriptors") || "[]"
      );

      // Add new descriptor
      existingDescriptors.push({
        label: newFaceDescriptor.label,
        descriptors: newFaceDescriptor.descriptors.map((d) => Array.from(d)),
      });

      // Save updated descriptors
      localStorage.setItem(
        "faceDescriptors",
        JSON.stringify(existingDescriptors)
      );

      // Update faceMatcher
      loadStoredDescriptors();
    }

    setIsTraining(false);
    setCapturedImages([]);
    alert("Training complete!");
  };

  const recognizeFace = async () => {
    if (!isModelLoaded || !webcamRef.current || !faceMatcher) return;

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);

    const detections = await faceapi
      .detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    resizedDetections.forEach((detection) => {
      const result = faceMatcher.findBestMatch(detection.descriptor);
      const box = detection.detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString(),
        lineWidth: 2,
        boxColor: "blue",
        drawLabelOptions: {
          fontSize: 16,
          fontStyle: "bold",
          fontColor: "white",
          fontFamily: "Arial",
          backgroundColor: "rgba(0, 0, 255, 0.8)",
        },
      });
      drawBox.draw(canvas);
    });

    if (resizedDetections.length > 0) {
      const result = faceMatcher.findBestMatch(resizedDetections[0].descriptor);
      setRecognizedName(result.toString());
    } else {
      setRecognizedName("No face detected");
    }
  };

  useEffect(() => {
    if (isModelLoaded && webcamRef.current) {
      const interval = setInterval(recognizeFace, 100);
      return () => clearInterval(interval);
    }
  }, [isModelLoaded, faceMatcher]);

  return (
    <div className="facial-recognition-app">
      <header>
        <h1>Facial Recognition App</h1>
      </header>
      {!isModelLoaded ? (
        <div className="loading">
          <p>Loading face recognition models...</p>
          <div className="spinner"></div>
        </div>
      ) : (
        <div className="app-content">
          <div className="webcam-container">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={640}
              height={480}
            />
            <canvas ref={canvasRef} className="face-overlay" />
          </div>
          <div className="controls">
            <button
              onClick={captureImage}
              disabled={isTraining}
              className={capturedImages.length >= 5 ? "complete" : ""}
            >
              Capture Image ({capturedImages.length}/5)
            </button>
            <button
              onClick={trainModel}
              disabled={isTraining || capturedImages.length < 5}
              className={isTraining ? "training" : ""}
            >
              {isTraining ? "Training..." : "Train Model"}
            </button>
          </div>
          <div className="captured-images">
            {capturedImages.map((src, index) => (
              <img
                key={index}
                src={src}
                alt={`Captured ${index + 1}`}
                className="captured-image"
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FacialRecognitionApp;
