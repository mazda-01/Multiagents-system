"use client"

import { useState } from "react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

export default function UiKitPage() {
  const [server, setServer] = useState("Localhost")
  const [host, setHost] = useState("127.0.0.1")
  const [port, setPort] = useState("5432")
  const [db, setDb] = useState("bookings")
  const [user, setUser] = useState("postgres")
  const [pwd, setPwd] = useState("")

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <div className="flex items-start justify-between gap-6">
        <div>
          <p className="tech-title text-xs text-muted-foreground">UI / Design System</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">SQL Agent — UI Kit</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Тут проверяем карточки/кнопки/модалки/табы. Потом переносим на главную.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="secondary">Server: {server}</Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setServer("Localhost")}>Localhost</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setServer("Staging")}>Staging</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setServer("Prod")}>Prod</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Dialog>
            <DialogTrigger asChild>
              <Button>Connect</Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[520px]">
              <DialogHeader>
                <DialogTitle>Подключение к PostgreSQL</DialogTitle>
                <DialogDescription>Пока UI-форма. В следующем шаге привяжем к backend.</DialogDescription>
              </DialogHeader>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Host</Label>
                  <Input value={host} onChange={(e) => setHost(e.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label>Port</Label>
                  <Input value={port} onChange={(e) => setPort(e.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label>Database</Label>
                  <Input value={db} onChange={(e) => setDb(e.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label>User</Label>
                  <Input value={user} onChange={(e) => setUser(e.target.value)} />
                </div>
                <div className="col-span-2 space-y-2">
                  <Label>Password</Label>
                  <Input type="password" value={pwd} onChange={(e) => setPwd(e.target.value)} />
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <Button variant="secondary" onClick={() => toast.message("Saved", { description: "Профиль подключения сохранён (пока локально)." })}>
                  Save
                </Button>
                <Button onClick={() => toast.success("Connected", { description: `${user}@${host}:${port}/${db}` })}>
                  Test connection
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <Separator className="my-8" />

      <div className="grid grid-cols-12 gap-6">
        <Card className="col-span-12 md:col-span-4">
          <CardHeader>
            <CardTitle>Query</CardTitle>
            <CardDescription>Отправка запроса в Text-to-SQL</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Textarea placeholder="Например: покажи топ-10 городов по числу вылетов" className="min-h-[140px]" />
            <Button className="w-full" onClick={() => toast("Run", { description: "Дальше привяжем к твоему /chat" })}>
              Execute
            </Button>
          </CardContent>
        </Card>

        <Card className="col-span-12 md:col-span-8">
          <CardHeader className="pb-3">
            <CardTitle>Output</CardTitle>
            <CardDescription>Таблица / график</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="table" className="w-full">
              <TabsList>
                <TabsTrigger value="table">Table</TabsTrigger>
                <TabsTrigger value="chart">Chart</TabsTrigger>
              </TabsList>
              <TabsContent value="table" className="mt-4">
                <div className="panel p-4 text-sm text-muted-foreground">
                  Тут будет DataGrid + кнопка Export CSV.
                </div>
              </TabsContent>
              <TabsContent value="chart" className="mt-4">
                <div className="panel p-4 text-sm text-muted-foreground">
                  Тут будет рендер base64 PNG + кнопка Export PNG.
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}