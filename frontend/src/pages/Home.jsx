import React from "react";
import { useNavigate } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className="bg-green-600 p-4 shadow-md">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <h1 className="text-white text-2xl font-bold">AgriWorld</h1>
        <ul className="flex space-x-4">
          <li>
            <a
              href="/"
              className="text-white hover:text-gray-200 transition-colors"
            >
              Home
            </a>
          </li>
          <li>
            <a
              href="/plant-disease"
              className="text-white hover:text-gray-200 transition-colors"
            >
              Plant Disease
            </a>
          </li>
          <li>
            <a
              href="/seedling"
              className="text-white hover:text-gray-200 transition-colors"
            >
              Seedling Prediction
            </a>
          </li>
        </ul>
      </div>
    </nav>
  );
};

const Home = () => {
  const navigate = useNavigate();

  return (
    <>
      <Navbar />
      <div className="bg-gray-100 min-h-screen flex flex-col items-center py-10">
        <h1 className="text-4xl font-bold mb-4 text-green-700">Home</h1>
        <p className="text-gray-700 text-lg text-center max-w-xl mb-8">
          Welcome to AgriWorld! You can get predictions about crop diseases or
          classify the type of seedling.
        </p>

        <div className="flex space-x-4">
          <button
            onClick={() => navigate("/plant-disease")}
            className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors"
          >
            Get Prediction
          </button>
          <button
            onClick={() => navigate("/seedling")}
            className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors"
          >
            Get Seedling
          </button>
        </div>
      </div>
    </>
  );
};

export default Home;
