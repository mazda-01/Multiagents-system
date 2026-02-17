import "./globals.css"
import { Toaster } from "@/components/ui/sonner"

export const metadata = {
  title: "SQL Agent",
  description: "Text-to-SQL agent UI",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru" className="dark" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground antialiased">
        <div className="hitech-bg">{children}</div>
        <Toaster richColors position="top-right" />
      </body>
    </html>
  )
}
