import "./App.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Seedling from "./pages/Seedling";
import { ImageProvider } from "./context/Base64Decode";
import PlantDisease from "./pages/PlantDisease";
import Home from "./pages/Home";

function App() {
  return (
    <ImageProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/seedling" element={<Seedling />} />
          <Route path="/plant-disease" element={<PlantDisease />} />
        </Routes>
      </BrowserRouter>
    </ImageProvider>
  );
}

export default App;
