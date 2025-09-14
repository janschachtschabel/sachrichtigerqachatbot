import React from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './index.css'
import App from './pages/App.jsx'
import Impressum from './pages/Impressum.jsx'

const router = createBrowserRouter([
  { path: '/', element: <App /> },
  { path: '/impressum', element: <Impressum /> },
])

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)
