import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Portfolio Agents — NVIDIA × SingleStore",
  description:
    "A fleet of GPU-accelerated strategy agents with truly persisted memory in SingleStore, competing on the S&P 500.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
