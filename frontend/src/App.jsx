import "./App.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Seedling from "./pages/Seedling";
import { ImageProvider } from "./context/Base64Decode";

function App() {
  return (
    <ImageProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/seedling" element={<Seedling />} />
        </Routes>
      </BrowserRouter>
    </ImageProvider>
  );
}

export default App;
