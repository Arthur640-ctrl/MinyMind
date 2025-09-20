import { Link } from 'react-router-dom'
import './Header.css'

export default function Header() {
  return (
    <header className="header">
      <div className="container">
        <div className="language-toggler">
          <span className="current-lang">EN</span>
          <i className="fa-solid fa-chevron-down"></i>
          <ul className="lang-dropdown">
            <li data-lang="fr">FR</li>
            <li data-lang="en">EN</li>
          </ul>
        </div>
        
        <nav>
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="/models">Models</Link></li>
            <li><Link to="/sandbox">SandBox</Link></li>
            <li><Link to="/pricing">Prices</Link></li>
            <li><Link to="/about">About</Link></li>
          </ul>
        </nav>
          
        <div className="auth-buttons">
          <button className="btn btn-outline">Login</button>
          <button className="btn btn-primary">Register</button>
        </div>
      </div>
    </header>
  )
}
