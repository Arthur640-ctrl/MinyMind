import './Register.css'

export default function Register() {
    return (
        <div className='main'>
            <div className="ambient-light light-1"></div>
            <div className="ambient-light light-2"></div>
            <div className="ambient-light light-3"></div>
            
            <div className="form">
                <p className="title">S'inscrire</p>
                <p className="message">Rejoignez l'expérience MinyMind et débloquez le plein potentiel de l'IA !</p>

                <div className="flex">
                    <label>
                        <input className="input" type="text" placeholder="" required />
                        <span>Prénom</span>
                    </label>

                    <label>
                        <input className="input" type="text" placeholder="" required />
                        <span>Nom</span>
                    </label>
                </div>  
                        
                <label>
                    <input className="input" type="email" placeholder="" required />
                    <span>Email</span>
                </label> 
                    
                <label>
                    <input className="input" type="password" placeholder="" required />
                    <span>Mot de Passe</span>
                </label>

                <label>
                    <input className="input" type="password" placeholder="" required />
                    <span>Confirmez le Mot de Passe</span>
                </label>

                <button className="submit">Créer un compte</button>
                <p className="signin">Déjà membre de NeuroSphere ? <a href="#">Se connecter</a> </p>
            </div>
        </div>
    )
}