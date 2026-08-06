import React, { useState, useRef } from "react";
import { QRCodeSVG } from "qrcode.react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/brand/atoms";
import { Copy, Printer, Check, QrCode } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

/* ── 4 Table Card Templates ── */
const TEMPLATES = [
  {
    id: "dark",
    name: "Modern Dark",
    bgClass: "bg-[#08090d] text-[#f1efe9] border border-[#262b38]",
    cardHeader: "text-[#8b7bff]",
    titleClass: "font-bold text-white",
    qrBg: "#ffffff",
    qrFg: "#08090d",
    accentDot: "bg-[#34d399]",
  },
  {
    id: "blush",
    name: "Minimalist Blush",
    bgClass: "bg-[#fdf6f5] text-[#2c1d27] border border-[#f3d9d7]",
    cardHeader: "text-[#d96b63]",
    titleClass: "font-serif text-[#2c1d27]",
    qrBg: "#ffffff",
    qrFg: "#2c1d27",
    accentDot: "bg-[#d96b63]",
  },
  {
    id: "botanical",
    name: "Botanical Greenery",
    bgClass: "bg-[#f4f7f4] text-[#1c3325] border border-[#d2e0d3]",
    cardHeader: "text-[#2e7d52]",
    titleClass: "font-serif text-[#1c3325]",
    qrBg: "#ffffff",
    qrFg: "#1c3325",
    accentDot: "bg-[#34d399]",
  },
  {
    id: "gold",
    name: "Art Deco Gold",
    bgClass: "bg-[#0b1329] text-[#f4e8c1] border-2 border-[#e8b04b]",
    cardHeader: "text-[#e8b04b]",
    titleClass: "font-serif text-[#f4e8c1] tracking-wider uppercase",
    qrBg: "#f4e8c1",
    qrFg: "#0b1329",
    accentDot: "bg-[#e8b04b]",
  },
];

export default function QrCardGeneratorModal({ open, onOpenChange, event }) {
  const [selectedTemplate, setSelectedTemplate] = useState(TEMPLATES[0]);
  const [copied, setCopied] = useState(false);
  const cardRef = useRef(null);

  if (!event) return null;

  const joinUrl = `${window.location.origin}/attendjoin?code=${event.event_code}`;

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(joinUrl);
      setCopied(true);
      toast.success("Event Join link copied to clipboard!");
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Failed to copy link.");
    }
  };

  const handlePrint = () => {
    const printWindow = window.open("", "_blank");
    if (!printWindow) return toast.error("Please allow popups to print.");

    const content = cardRef.current ? cardRef.current.outerHTML : "";
    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>Print QR Table Card - ${event.title}</title>
          <script src="https://cdn.tailwindcss.com"></script>
          <style>
            @media print {
              body { margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
            }
          </style>
        </head>
        <body class="bg-white flex items-center justify-center p-8">
          ${content}
          <script>
            setTimeout(() => { window.print(); window.close(); }, 500);
          </script>
        </body>
      </html>
    `);
    printWindow.document.close();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="border-frame bg-surface text-text sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl font-semibold tracking-tight">
            <QrCode className="h-5 w-5 text-iris-2" /> QR Table Card Generator
          </DialogTitle>
        </DialogHeader>

        {/* Template Selector */}
        <div className="mt-2 space-y-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-text-faint mb-2">
              Select Design Template
            </label>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {TEMPLATES.map((tmpl) => (
                <button
                  key={tmpl.id}
                  onClick={() => setSelectedTemplate(tmpl)}
                  className={cn(
                    "flex flex-col items-center justify-center rounded-xl p-3 text-xs font-medium border transition-all",
                    selectedTemplate.id === tmpl.id
                      ? "border-iris-2 bg-iris/10 text-text"
                      : "border-frame-soft bg-surface-2 text-text-dim hover:border-frame"
                  )}
                >
                  <span className={cn("mb-1 h-2 w-2 rounded-full", tmpl.accentDot)} />
                  {tmpl.name}
                </button>
              ))}
            </div>
          </div>

          {/* Printable Card Live Preview */}
          <div className="flex justify-center p-4 rounded-xl bg-ink/60 border border-frame-soft">
            <div
              ref={cardRef}
              className={cn(
                "relative w-72 rounded-2xl p-6 text-center shadow-2xl flex flex-col items-center justify-between min-h-[380px]",
                selectedTemplate.bgClass
              )}
            >
              {/* Header */}
              <div className="w-full">
                <p className={cn("text-[0.7rem] font-mono uppercase tracking-[0.18em]", selectedTemplate.cardHeader)}>
                  Share Your Memories
                </p>
                <h3 className={cn("mt-1 text-xl font-bold line-clamp-1", selectedTemplate.titleClass)}>
                  {event.title}
                </h3>
                <p className="mt-1 text-xs opacity-75">
                  Scan QR to upload or view photos
                </p>
              </div>

              {/* QR Code Container */}
              <div className="my-4 p-3 rounded-xl bg-white shadow-md inline-block">
                <QRCodeSVG
                  value={joinUrl}
                  size={140}
                  bgColor={selectedTemplate.qrBg}
                  fgColor={selectedTemplate.qrFg}
                  level="H"
                  includeMargin={false}
                />
              </div>

              {/* Footer Code */}
              <div className="w-full pt-2 border-t border-current/10">
                <p className="text-[0.65rem] opacity-60 uppercase tracking-widest">Access Code</p>
                <p className="font-mono text-lg font-bold tracking-widest mt-0.5">
                  {event.event_code}
                </p>
              </div>
            </div>
          </div>

          {/* Action Controls */}
          <div className="flex flex-col sm:flex-row gap-3 pt-2">
            <Button variant="primary" onClick={handlePrint} className="flex-1 gap-2">
              <Printer className="h-4 w-4" /> Print / Save PDF
            </Button>
            <Button variant="subtle" onClick={handleCopyLink} className="flex-1 gap-2">
              {copied ? <Check className="h-4 w-4 text-match" /> : <Copy className="h-4 w-4" />}
              {copied ? "Link Copied!" : "Copy Direct Link"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
