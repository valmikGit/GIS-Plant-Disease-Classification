import "./App.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Seedling from "./pages/Seedling";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/seedling" element={<Seedling />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
