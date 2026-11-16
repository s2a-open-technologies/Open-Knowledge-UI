import { StrictMode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "@/App";
import { AuthProvider } from "@/context/AuthContext";
import { IngestionProvider } from "@/context/IngestionContext";
import { ThemeProvider } from "@/context/ThemeContext";
import "@fontsource-variable/inter";
import "@/index.css";

// Eine Anfrage nicht automatisch wiederholen: Fehler sollen sofort als Meldung erscheinen.
const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <IngestionProvider>
            <App />
          </IngestionProvider>
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
