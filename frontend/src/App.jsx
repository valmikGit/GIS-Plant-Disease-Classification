import "./App.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Seedling from "./pages/Seedling";
import { ImageProvider } from "./context/Base64Decode";
import PlantDisease from "./pages/PlantDisease";

function App() {
  return (
    <ImageProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/seedling" element={<Seedling />} />
          <Route path="/plant-disease" element={<PlantDisease />} />
        </Routes>
      </BrowserRouter>
    </ImageProvider>
  );
}

export default App;
