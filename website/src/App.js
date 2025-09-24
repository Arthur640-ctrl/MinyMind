import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Home from './pages/Home'
import './App.css'
import Register from './pages/Register'
import LayoutWithHeader from './layouts/LayoutWithHeader'
import LayoutWithoutHeader from './layouts/LayoutWithoutHeader'



function App() {
  return (
    <Router>
      <div className="App">
        <div className="ambient-light light-1"></div>
        <div className="ambient-light light-2"></div>
        <div className="ambient-light light-3"></div>

        <Routes>
          {/* Avec header */}
          <Route element={<LayoutWithHeader />}>
            <Route path="/" element={<Home />} />
          </Route>

          {/* Sans header */}
          <Route element={<LayoutWithoutHeader />}>
            <Route path="/register" element={<Register />} />
          </Route>
        </Routes>
      </div>
    </Router>
  )
}

export default App;
