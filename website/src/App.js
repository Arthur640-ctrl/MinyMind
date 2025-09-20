import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Home from './pages/Home'
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <div className="ambient-light light-1"></div>
        <div className="ambient-light light-2"></div>
        <div className="ambient-light light-3"></div>
        <Header />
        
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App;
