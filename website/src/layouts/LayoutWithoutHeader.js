import { Outlet } from "react-router-dom"

export default function LayoutWithoutHeader() {
  return (
    <>
      <Outlet />
    </>
  )
}
