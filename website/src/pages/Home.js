import './Home.css'

// const LinkWithLabel = ({ x1, y1, x2, y2, label }) => (
//   <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}>
//     <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="black" strokeWidth="2" />
//     <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 5} textAnchor="middle" fontWeight="bold">
//       {label}
//     </text>
//   </svg>
// );

export default function Home() {
  return (
    <section className="hero">
      <div className="container">
          <div className="hero-content">
              <h1>MinyMind - L'IA de A à Z.</h1>
              <h1>Une Experience Ultime.</h1>

              <p>Utilisez nos modèles ou utilisez les vôtres gratuitement dans une interface conviviale et interactive</p>
              <div className="hero-buttons">
                  <button className="btn btn-primary">Explorer les modèles</button>
                  <button className="btn btn-outline">Voir la démo</button>
              </div>
          </div>
      </div>
        
      {/* <div style={{ position: "absolute", left: 100, top: 100 }}>O</div>
      <div style={{ position: "absolute", left: 400, top: 200 }}>O</div> */}

      {/* Ligne avec texte */}
      {/* <LinkWithLabel x1={100} y1={100} x2={400} y2={200} label="26" /> */}
    </section>

  )
}