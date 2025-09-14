import React from 'react'
import { Link } from 'react-router-dom'

export default function Impressum() {
  return (
    <div className="min-h-screen bg-bg text-text">
      <header className="border-b border-slate-700/50 bg-panel">
        <div className="mx-auto max-w-5xl px-4 py-3 flex items-center justify-between">
          <h1 className="text-lg font-semibold">Sachrichtiger QA-Chatbot</h1>
          <nav className="text-sm text-muted flex items-center gap-4">
            <Link className="hover:text-text" to="/">Chat</Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-3xl px-4 py-8 space-y-6">
        <h2 className="text-2xl font-bold">Impressum</h2>
        <div className="space-y-1">
          <p>Jan Schachtschabel</p>
          <p>Steubenstr. 34</p>
          <p>99423 Weimar</p>
        </div>
        <div className="space-y-2">
          <h3 className="text-xl font-semibold">Kontakt</h3>
          <p>E-Mail: <a className="text-primary hover:underline" href="mailto:jan@schachtschabel.net">jan@schachtschabel.net</a></p>
          <form className="mt-3" action="mailto:jan@schachtschabel.net" method="post" encType="text/plain">
            <div className="grid gap-2 sm:grid-cols-2">
              <input name="name" required placeholder="Ihr Name" className="rounded-md bg-[#0b0f18] border border-slate-700/60 px-3 py-2" />
              <input name="email" type="email" required placeholder="Ihre E-Mail" className="rounded-md bg-[#0b0f18] border border-slate-700/60 px-3 py-2" />
            </div>
            <textarea name="nachricht" required placeholder="Ihre Nachricht" rows="5" className="mt-2 w-full rounded-md bg-[#0b0f18] border border-slate-700/60 px-3 py-2"></textarea>
            <button type="submit" className="mt-3 inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-white hover:brightness-110">Senden</button>
          </form>
        </div>
      </main>
      <footer className="border-t border-slate-700/50 bg-panel">
        <div className="mx-auto max-w-5xl px-4 py-3 text-sm text-muted">
          © {new Date().getFullYear()} Jan Schachtschabel
        </div>
      </footer>
    </div>
  )
}
