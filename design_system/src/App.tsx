import React, { useState } from 'react';
import { Button } from './components/ui/Button';
import { Card } from './components/ui/Card';
import { ShinyBadge } from './components/ui/ShinyBadge';
import { Input, FormField } from './components/ui/FormField';
import { ShieldCheck, Cpu, Terminal, Key, Sparkles, CheckCircle2, RefreshCw } from 'lucide-react';

export const App: React.FC = () => {
  const [apiKey, setApiKey] = useState('mb_live_992831018a7c4e2');
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(apiKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-surface-canvas text-content-primary p-6 md:p-12 font-sans">
      <div className="max-w-5xl mx-auto space-y-10">
        
        {/* Header Bar */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-line-default pb-8">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-content-primary">
                Developer Integration Hub
              </h1>
              <ShinyBadge variant="iridescent" pulse icon={<Sparkles className="w-3.5 h-3.5" />}>
                v2.4 Live
              </ShinyBadge>
            </div>
            <p className="text-sm text-content-secondary mt-1.5 max-w-xl">
              Clean, developer-friendly theme inspired by Metabase&apos;s slate surfaces, crisp typography, and signature royal blue accents.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button variant="outline" size="md" icon={<Terminal className="w-4 h-4" />}>
              CLI Reference
            </Button>
            <Button variant="primary" size="md" icon={<Cpu className="w-4 h-4" />}>
              Deploy Connector
            </Button>
          </div>
        </header>

        {/* Shiny Badge Variants Section */}
        <section className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-content-muted">
            Shiny Badge Showcase
          </h2>
          <div className="flex flex-wrap items-center gap-3 p-4 bg-white rounded-xl border border-line-default shadow-crisp-sm">
            <ShinyBadge variant="iridescent" pulse icon={<Sparkles className="w-3.5 h-3.5" />}>
              Iridescent Shimmer
            </ShinyBadge>
            <ShinyBadge variant="blue-glow" pulse icon={<ShieldCheck className="w-3.5 h-3.5" />}>
              Royal Blue Accent
            </ShinyBadge>
            <ShinyBadge variant="emerald" pulse fontMono icon={<CheckCircle2 className="w-3.5 h-3.5" />}>
              99.99% Uptime
            </ShinyBadge>
            <ShinyBadge variant="amber" pulse icon={<Key className="w-3.5 h-3.5" />}>
              Token Expiring
            </ShinyBadge>
          </div>
        </section>

        {/* Core Component Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Card 1: API Config */}
          <Card
            title="Embedded Analytics SDK"
            subtitle="Client-side token authentication & security"
            badge={<ShinyBadge variant="blue-glow">Pro Feature</ShinyBadge>}
            interactive
            footer={
              <div className="flex items-center justify-between text-xs text-content-muted">
                <span>Last updated 2 hours ago</span>
                <a href="#docs" className="text-brand-600 hover:text-brand-700 font-semibold inline-flex items-center gap-1">
                  Documentation &rarr;
                </a>
              </div>
            }
          >
            <div className="space-y-4">
              <FormField label="API Secret Token" hint="Keep private" required>
                <div className="flex gap-2">
                  <Input 
                    value={apiKey} 
                    onChange={(e) => setApiKey(e.target.value)} 
                    placeholder="Enter token key..." 
                  />
                  <Button variant="secondary" size="md" onClick={handleCopy}>
                    {copied ? 'Copied!' : 'Copy'}
                  </Button>
                </div>
              </FormField>

              <FormField label="Allowed CORS Domains" hint="Comma separated">
                <Input defaultValue="https://analytics.internal, https://app.example.com" />
              </FormField>
              
              <div className="flex gap-2 pt-2">
                <Button variant="primary" size="sm" icon={<RefreshCw className="w-3.5 h-3.5" />}>
                  Rotate Key
                </Button>
                <Button variant="outline" size="sm">
                  Test Sandbox
                </Button>
                <Button variant="ghost" size="sm">
                  Revoke
                </Button>
              </div>
            </div>
          </Card>

          {/* Card 2: Query Telemetry */}
          <Card
            title="Database Telemetry Engine"
            subtitle="Real-time SQL performance & connection pooling"
            badge={<ShinyBadge variant="emerald" pulse fontMono>PostgreSQL Connected</ShinyBadge>}
            interactive
            footer={
              <div className="flex items-center justify-between text-xs text-content-muted">
                <span>Engine latency: <strong className="text-slate-700">12ms</strong></span>
                <span className="text-emerald-600 font-semibold flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" /> Live Pool
                </span>
              </div>
            }
          >
            <div className="space-y-4">
              <p className="text-sm text-content-secondary leading-relaxed">
                Schemas are synchronized automatically every 15 minutes to guarantee crisp, zero-latency dashboard renders.
              </p>
              
              <div className="p-3.5 bg-slate-900 rounded-lg font-mono text-xs text-slate-200 overflow-x-auto shadow-inner space-y-1">
                <p className="text-slate-400">// Sample automated query trace</p>
                <p className="text-brand-300">SELECT <span className="text-purple-300">count</span>(*), <span className="text-emerald-300">date_trunc</span>(&apos;day&apos;, created_at)</p>
                <p className="text-brand-300">FROM <span className="text-amber-300">analytics_events</span></p>
                <p className="text-brand-300">WHERE <span className="text-slate-300">status = &apos;active&apos;</span></p>
                <p className="text-brand-300">GROUP BY <span className="text-slate-300">2</span> ORDER BY <span className="text-slate-300">2 DESC</span>;</p>
              </div>

              <div className="flex items-center justify-between pt-1">
                <span className="text-xs text-content-muted">Execution speed: <strong className="text-slate-800">4.2ms avg</strong></span>
                <Button variant="outline" size="sm">
                  View Full Logs
                </Button>
              </div>
            </div>
          </Card>

        </div>

      </div>
    </div>
  );
};

export default App;
