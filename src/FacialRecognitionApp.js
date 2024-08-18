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
  const [activeTab, setActiveTab] = useState("train");
  const [faceDetected, setFaceDetected] = useState(false);
  const [facePosition, setFacePosition] = useState("not-detected");
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [location, setLocation] = useState(null);
  const [isGettingLocation, setIsGettingLocation] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (isModelLoaded) {
      loadStoredDescriptors();
    }
  }, [isModelLoaded]);

  useEffect(() => {
    resetState();
  }, [activeTab]);

  const resetState = () => {
    setIsVideoReady(false);
    setFaceDetected(false);
    setFacePosition("not-detected");
    setRecognizedName("");
    setLocation(null);
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const loadModels = async () => {
    const MODEL_URL =
      "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights";

    try {
      await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
      setIsModelLoaded(true);
      console.log("Face recognition models loaded successfully");
    } catch (error) {
      console.error("Error loading models:", error);
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

  const detectFace = async () => {
    if (!webcamRef.current || !canvasRef.current || !isVideoReady) return;

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log("Video dimensions not ready yet");
      return;
    }

    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(canvas, displaySize);

    try {
      const detections = await faceapi
        .detectAllFaces(video)
        .withFaceLandmarks();
      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

      if (resizedDetections.length > 0) {
        setFaceDetected(true);
        const detection = resizedDetections[0];
        const box = detection.detection.box;

        if (activeTab === "train" && isValidBox(box)) {
          const drawBox = new faceapi.draw.DrawBox(box, {
            label: getFacePositionLabel(box, displaySize),
            boxColor: getFacePositionColor(box, displaySize),
          });
          drawBox.draw(canvas);
          setFacePosition(getFacePositionLabel(box, displaySize));
        }
      } else {
        setFaceDetected(false);
        setFacePosition("not-detected");
      }
    } catch (error) {
      console.error("Error in detectFace:", error);
    }
  };

  const isValidBox = (box) => {
    return (
      box &&
      typeof box.x === "number" &&
      typeof box.y === "number" &&
      typeof box.width === "number" &&
      typeof box.height === "number"
    );
  };

  const getFacePositionLabel = (box, displaySize) => {
    if (!isValidBox(box)) return "Invalid face position";

    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    const threshold = 0.2;

    if (
      box.width < displaySize.width * 0.3 ||
      box.height < displaySize.height * 0.3
    ) {
      return "Move closer";
    }

    if (
      box.width > displaySize.width * 0.8 ||
      box.height > displaySize.height * 0.8
    ) {
      return "Move farther";
    }

    if (centerX < displaySize.width * threshold) return "Move right";
    if (centerX > displaySize.width * (1 - threshold)) return "Move left";
    if (centerY < displaySize.height * threshold) return "Move down";
    if (centerY > displaySize.height * (1 - threshold)) return "Move up";

    return "Good position";
  };

  const getFacePositionColor = (box, displaySize) => {
    const label = getFacePositionLabel(box, displaySize);
    return label === "Good position" ? "green" : "red";
  };

  useEffect(() => {
    let intervalId;
    if (isModelLoaded && webcamRef.current && isVideoReady) {
      const processFace = () => {
        if (activeTab === "train") {
          detectFace();
        } else {
          recognizeFace();
        }
      };

      intervalId = setInterval(processFace, 100);
    }
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isModelLoaded, activeTab, isVideoReady]);

  const getLocation = async () => {
    return new Promise((resolve, reject) => {
      if ("geolocation" in navigator) {
        setIsGettingLocation(true);
        navigator.geolocation.getCurrentPosition(
          async (position) => {
            try {
              const { latitude, longitude } = position.coords;
              const response = await fetch(
                `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&zoom=18&addressdetails=1`
              );
              const data = await response.json();
              setIsGettingLocation(false);
              resolve({
                latitude,
                longitude,
                city:
                  data.address.city ||
                  data.address.town ||
                  data.address.village,
                state: data.address.state,
                country: data.address.country,
                postcode: data.address.postcode,
                fullAddress: data.display_name,
              });
            } catch (error) {
              console.error("Error fetching location details:", error);
              setIsGettingLocation(false);
              reject(error);
            }
          },
          (error) => {
            setIsGettingLocation(false);
            reject(error);
          }
        );
      } else {
        reject(new Error("Geolocation is not supported by this browser."));
      }
    });
  };

  const captureImage = async () => {
    if (webcamRef.current && faceDetected && facePosition === "Good position") {
      try {
        const imageSrc = webcamRef.current.getScreenshot();
        const currentLocation = await getLocation();
        setCapturedImages((prev) => [
          ...prev,
          { src: imageSrc, location: currentLocation },
        ]);
        setLocation(currentLocation);
      } catch (error) {
        console.error("Error getting location:", error);
        alert("Failed to get location. Image captured without location data.");
        const imageSrc = webcamRef.current.getScreenshot();
        setCapturedImages((prev) => [
          ...prev,
          { src: imageSrc, location: null },
        ]);
      }
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
      const img = await faceapi.fetchImage(capturedImages[i].src);
      const detection = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        descriptions.push({
          descriptor: detection.descriptor,
          location: capturedImages[i].location,
        });
      }
    }

    if (descriptions.length > 0) {
      const newFaceDescriptor = new faceapi.LabeledFaceDescriptors(
        personName,
        descriptions.map((d) => d.descriptor)
      );
      const existingDescriptors = JSON.parse(
        localStorage.getItem("faceDescriptors") || "[]"
      );
      existingDescriptors.push({
        label: newFaceDescriptor.label,
        descriptors: newFaceDescriptor.descriptors.map((d) => Array.from(d)),
        locations: descriptions.map((d) => d.location),
      });
      localStorage.setItem(
        "faceDescriptors",
        JSON.stringify(existingDescriptors)
      );
      loadStoredDescriptors();
    }

    setIsTraining(false);
    setCapturedImages([]);
    alert("Training complete!");
  };

  const recognizeFace = async () => {
    if (!isModelLoaded || !webcamRef.current || !faceMatcher || !isVideoReady)
      return;

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log("Video dimensions not ready yet");
      return;
    }

    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(canvas, displaySize);

    try {
      const detections = await faceapi
        .detectAllFaces(video)
        .withFaceLandmarks()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

      if (resizedDetections.length === 0) {
        setRecognizedName("Person not detected");
        setLocation(null);
        return;
      }

      let recognizedPerson = null;

      resizedDetections.forEach((detection) => {
        const result = faceMatcher.findBestMatch(detection.descriptor);
        const box = detection.detection.box;
        if (isValidBox(box)) {
          const drawBox = new faceapi.draw.DrawBox(box, {
            label: result.label !== "unknown" ? result.label : "Unknown person",
            lineWidth: 2,
            boxColor: result.label !== "unknown" ? "blue" : "red",
            drawLabelOptions: {
              fontSize: 16,
              fontStyle: "bold",
              fontColor: "white",
              backgroundColor:
                result.label !== "unknown"
                  ? "rgba(0, 0, 255, 0.8)"
                  : "rgba(255, 0, 0, 0.8)",
            },
          });
          drawBox.draw(canvas);

          if (result.label !== "unknown") {
            recognizedPerson = result;
          }
        }
      });

      if (recognizedPerson) {
        setRecognizedName(recognizedPerson.label);
        try {
          const currentLocation = await getLocation();
          setLocation(currentLocation);
        } catch (error) {
          console.error("Error getting location:", error);
          setLocation(null);
        }
      } else {
        setRecognizedName("Unknown person");
        setLocation(null);
      }
    } catch (error) {
      console.error("Error in recognizeFace:", error);
      setRecognizedName("Error detecting face");
      setLocation(null);
    }
  };

  const handleVideoLoad = () => {
    setIsVideoReady(true);
  };

  const renderTrainingTab = () => (
    <div className="training-tab">
      <h2>Training</h2>
      <div className="training-steps">
        <h3>Steps to Train the Model:</h3>
        <ol>
          <li>Ensure good lighting and a clear background.</li>
          <li>Position your face in the center of the camera view.</li>
          <li>
            Follow the on-screen instructions to properly position your face.
          </li>
          <li>Click "Capture Image" when your face is in a good position.</li>
          <li>
            Capture 5 different images with varied expressions and angles.
          </li>
          <li>Once you've captured 5 images, click "Train Model".</li>
          <li>Enter your name when prompted to complete the training.</li>
        </ol>
      </div>
      <div className="webcam-container">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={{ width: 640, height: 480 }}
          onLoadedData={handleVideoLoad}
        />
        <canvas ref={canvasRef} className="face-overlay" />
        <div className="face-position-label">{facePosition}</div>
      </div>
      <div className="controls">
        <button
          onClick={captureImage}
          disabled={
            isTraining ||
            !faceDetected ||
            facePosition !== "Good position" ||
            isGettingLocation
          }
          className={capturedImages.length >= 5 ? "complete" : ""}
        >
          {isGettingLocation
            ? "Getting Location..."
            : `Capture Image (${capturedImages.length}/5)`}
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
        {capturedImages.map((image, index) => (
          <div key={index} className="captured-image-container">
            <img
              src={image.src}
              alt={`Captured ${index + 1}`}
              className="captured-image"
            />
            {image.location && (
              <div className="image-location">
                <p>
                  Lat: {image.location.latitude.toFixed(4)}, Lon:{" "}
                  {image.location.longitude.toFixed(4)}
                </p>
                <p>
                  {image.location.city}, {image.location.state},{" "}
                  {image.location.country}
                </p>
                <p>{image.location.fullAddress}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const renderTestingTab = () => (
    <div className="testing-tab">
      <h2>Testing</h2>
      <div className="webcam-container">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={{ width: 640, height: 480 }}
          onLoadedData={handleVideoLoad}
        />
        <canvas ref={canvasRef} className="face-overlay" />
      </div>
      <div className="recognition-result">
        <h3>Recognition Result:</h3>
        <p>{recognizedName}</p>
        {location && (
          <div className="current-location">
            <p>Current Location:</p>
            <p>
              Lat: {location.latitude.toFixed(4)}, Lon:{" "}
              {location.longitude.toFixed(4)}
            </p>
            <p>
              {location.city}, {location.state}, {location.country}
            </p>
            <p>{location.fullAddress}</p>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="facial-recognition-app">
      <header>
        <h1>Facial Recognition System with Geolocation</h1>
      </header>
      {!isModelLoaded ? (
        <div className="loading">
          <p>Loading face recognition models...</p>
          <div className="spinner"></div>
        </div>
      ) : (
        <div className="app-content">
          <div className="tabs">
            <button
              className={activeTab === "train" ? "active" : ""}
              onClick={() => setActiveTab("train")}
            >
              Training
            </button>
            <button
              className={activeTab === "test" ? "active" : ""}
              onClick={() => setActiveTab("test")}
            >
              Testing
            </button>
          </div>
          {activeTab === "train" ? renderTrainingTab() : renderTestingTab()}
        </div>
      )}
    </div>
  );
};

export default FacialRecognitionApp;
