import './Home.css'

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
        <div id="particle-canvas"></div>
    </section>

  )
}