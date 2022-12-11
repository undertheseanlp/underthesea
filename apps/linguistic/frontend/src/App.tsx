import React from 'react'
import './App.css'
import 'bootstrap/dist/css/bootstrap.min.css'
import { NavbarMain } from './components/NavbarMain'
import { Article } from './components/Article'

function App() {
  return (
    <div>
      <NavbarMain></NavbarMain>
      <Article></Article>
    </div>
  )
}

export default App
