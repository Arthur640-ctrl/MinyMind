import Header from "../components/Header"
import { Outlet } from "react-router-dom"

export default function LayoutWithHeader() {
  return (
    <>
      <Header />
      <Outlet /> {/* Ici s’affiche la page */}
    </>
  )
}
