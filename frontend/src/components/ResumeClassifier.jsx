// src/components/ResumeClassifier.jsx
import { useState } from "react";
import axios from "axios";

export default function ResumeClassifier() {
  const [resumeText, setResumeText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    try {
      const response = await axios.post("http://localhost:8000/predict", {
        resume_text: resumeText,
      });
      setResult(response.data);
      setError("");
    } catch (err) {
      console.error(err);
      setError("Something went wrong while connecting to the API.");
      setResult(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center px-4 py-8">
      <div className="bg-white p-8 rounded-2xl shadow-md w-full max-w-3xl">
        <h1 className="text-2xl font-bold mb-6 text-center text-blue-700">
          Resume Categorization Assistant
        </h1>

        <textarea
          className="w-full h-48 p-4 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
          placeholder="Paste your resume text here..."
          value={resumeText}
          onChange={(e) => setResumeText(e.target.value)}
        ></textarea>

        <button
          className="mt-4 w-full bg-blue-600 text-white font-semibold py-3 rounded-lg hover:bg-blue-700 transition"
          onClick={handleSubmit}
        >
          Categorize Resume
        </button>

        {error && (
          <p className="text-red-600 mt-4 text-center">{error}</p>
        )}

        {result && (
          <div className="mt-6 border-t pt-4 space-y-2">
            <p className="text-lg font-semibold">
              📌 Predicted Role:{" "}
              <span className="text-blue-600">{result.predicted_category}</span>
            </p>
            <p className="text-md">
              ✅ Confidence Score:{" "}
              <span className="text-green-600">{(result.confidence_score * 100).toFixed(2)}%</span>
            </p>
            <div>
              <p className="font-semibold mb-1">🔍 Top Similar Roles:</p>
              <ul className="list-disc list-inside text-gray-700">
                {(result.similar_roles || []).map((item, idx) => (
  <li key={idx}>
    <span className="font-medium">{item.job_role}:</span>{" "}
    {(item.similarity_score * 100).toFixed(2)}%
  </li>
))}

              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
