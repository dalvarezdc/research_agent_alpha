"use client";

import React, { useEffect, useRef } from "react";

export interface CloudShaderProps extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
  speed?: number;
  density?: number;
  scale?: number;
  contrast?: number;
  brightness?: number;
  interactive?: boolean;
  children?: React.ReactNode;
}

const VERTEX_SHADER_SOURCE = `
  attribute vec2 position;
  varying vec2 vUv;
  void main() {
    vUv = (position + 1.0) * 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;

const FRAGMENT_SHADER_SOURCE = `
  precision highp float;
  
  uniform vec2 u_resolution;
  uniform float u_time;
  uniform vec2 u_mouse;
  uniform float u_speed;
  uniform float u_density;
  uniform float u_scale;
  uniform float u_contrast;
  uniform float u_brightness;
  
  varying vec2 vUv;

  // Hash function for pseudo-random 2D noise
  vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
  }

  // Simplex-style smooth gradient noise
  float noise(vec2 p) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2
    const float K2 = 0.211324865; // (3-sqrt(3))/6

    vec2 i = floor(p + (p.x + p.y) * K1);
    vec2 a = p - i + (i.x + i.y) * K2;
    float m = step(a.y, a.x); 
    vec2 o = vec2(m, 1.0 - m);
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;

    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    vec3 n = h * h * h * h * vec3(dot(a, hash2(i)), dot(b, hash2(i + o)), dot(c, hash2(i + 1.0)));

    return dot(n, vec3(70.0));
  }

  // Fractional Brownian Motion (fBm) for organic cloud layering
  float fbm(vec2 p) {
    float f = 0.0;
    mat2 rot = mat2(1.6, 1.2, -1.2, 1.6);
    f += 0.5000 * noise(p); p = rot * p;
    f += 0.2500 * noise(p); p = rot * p;
    f += 0.1250 * noise(p); p = rot * p;
    f += 0.0625 * noise(p); p = rot * p;
    f += 0.03125 * noise(p);
    return f;
  }

  void main() {
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    st.x *= u_resolution.x / u_resolution.y;

    float t = u_time * u_speed * 0.12;
    vec2 mouseOffset = (u_mouse - 0.5) * 0.4;
    
    vec2 uv = st * u_scale + mouseOffset;
    
    // Multi-layered wind drift simulation
    vec2 q = vec2(
      fbm(uv + vec2(t * 0.4, t * 0.1)),
      fbm(uv + vec2(t * 0.2, t * 0.3) + vec2(5.2, 1.3))
    );

    vec2 r = vec2(
      fbm(uv + 3.0 * q + vec2(1.7, 9.2) + 0.15 * t),
      fbm(uv + 3.0 * q + vec2(8.3, 2.8) + 0.126 * t)
    );

    float f = fbm(uv + 3.5 * r + vec2(t * 0.1, t * 0.05));

    // Cloud density shaping and thresholding
    f = (f * 0.5 + 0.5);
    f = pow(f * u_density, u_contrast) * u_brightness;

    // Palette Colors: Royal Slate / Ethereal Sky to Crisp Cumulus White
    vec3 skyTop = vec3(0.08, 0.18, 0.38);       // Deep slate navy
    vec3 skyBottom = vec3(0.24, 0.46, 0.78);    // Signature royal cerulean
    vec3 cloudShadow = vec3(0.42, 0.55, 0.72);  // Atmospheric shadow
    vec3 cloudMid = vec3(0.85, 0.91, 0.98);     // Soft billow light
    vec3 cloudHighlight = vec3(1.0, 1.0, 1.0);  // Pure sunlight crest

    vec3 skyColor = mix(skyTop, skyBottom, vUv.y);

    // Blend cloud body and highlights
    vec3 cloudColor = mix(cloudShadow, cloudMid, clamp(f * 1.4, 0.0, 1.0));
    cloudColor = mix(cloudColor, cloudHighlight, clamp(pow(f, 2.5), 0.0, 1.0));

    // Composite cloud over sky atmosphere
    float alpha = clamp((f - 0.15) * 1.5, 0.0, 1.0);
    vec3 finalColor = mix(skyColor, cloudColor, alpha);

    // Subtle edge vignette
    vec2 center = vUv - 0.5;
    float vignette = 1.0 - dot(center, center) * 0.35;
    finalColor *= vignette;

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

function createShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compilation error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export function CloudShader({
  className = "",
  speed = 1.0,
  density = 1.2,
  scale = 2.2,
  contrast = 1.6,
  brightness = 1.1,
  interactive = true,
  children,
  ...props
}: CloudShaderProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef<{ x: number; y: number; targetX: number; targetY: number }>({
    x: 0.5,
    y: 0.5,
    targetX: 0.5,
    targetY: 0.5,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const gl = canvas.getContext("webgl", {
      alpha: false,
      antialias: true,
      powerPreference: "high-performance",
    });

    if (!gl) {
      console.warn("WebGL not supported on this device/browser.");
      return;
    }

    const vertShader = createShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
    const fragShader = createShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);
    if (!vertShader || !fragShader) return;

    const program = gl.createProgram();
    if (!program) return;

    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("WebGL program link error:", gl.getProgramInfoLog(program));
      return;
    }

    gl.useProgram(program);

    // Full screen quad geometry
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -1, -1,
         1, -1,
        -1,  1,
        -1,  1,
         1, -1,
         1,  1,
      ]),
      gl.STATIC_DRAW
    );

    const positionLocation = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    // Uniform locations
    const uResolutionLoc = gl.getUniformLocation(program, "u_resolution");
    const uTimeLoc = gl.getUniformLocation(program, "u_time");
    const uMouseLoc = gl.getUniformLocation(program, "u_mouse");
    const uSpeedLoc = gl.getUniformLocation(program, "u_speed");
    const uDensityLoc = gl.getUniformLocation(program, "u_density");
    const uScaleLoc = gl.getUniformLocation(program, "u_scale");
    const uContrastLoc = gl.getUniformLocation(program, "u_contrast");
    const uBrightnessLoc = gl.getUniformLocation(program, "u_brightness");

    // Dynamic props uniforms
    gl.uniform1f(uSpeedLoc, speed);
    gl.uniform1f(uDensityLoc, density);
    gl.uniform1f(uScaleLoc, scale);
    gl.uniform1f(uContrastLoc, contrast);
    gl.uniform1f(uBrightnessLoc, brightness);

    let animationFrameId: number;
    const startTime = performance.now();

    const resize = () => {
      const rect = container.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.floor(rect.width * dpr);
      const height = Math.floor(rect.height * dpr);

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = Math.max(width, 1);
        canvas.height = Math.max(height, 1);
        gl.viewport(0, 0, canvas.width, canvas.height);
        if (uResolutionLoc) {
          gl.uniform2f(uResolutionLoc, canvas.width, canvas.height);
        }
      }
    };

    const resizeObserver = new ResizeObserver(() => resize());
    resizeObserver.observe(container);
    resize();

    const handleMouseMove = (e: MouseEvent) => {
      if (!interactive) return;
      const rect = container.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        const nx = (e.clientX - rect.left) / rect.width;
        const ny = 1.0 - (e.clientY - rect.top) / rect.height;
        mouseRef.current.targetX = Math.max(0, Math.min(1, nx));
        mouseRef.current.targetY = Math.max(0, Math.min(1, ny));
      }
    };

    if (interactive) {
      window.addEventListener("mousemove", handleMouseMove, { passive: true });
    }

    const render = (currentTime: number) => {
      // Smooth mouse lerping
      const mouse = mouseRef.current;
      mouse.x += (mouse.targetX - mouse.x) * 0.05;
      mouse.y += (mouse.targetY - mouse.y) * 0.05;

      const elapsed = (currentTime - startTime) * 0.001;
      gl.uniform1f(uTimeLoc, elapsed);
      gl.uniform2f(uMouseLoc, mouse.x, mouse.y);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      if (interactive) {
        window.removeEventListener("mousemove", handleMouseMove);
      }
      if (positionBuffer) gl.deleteBuffer(positionBuffer);
      if (vertShader) gl.deleteShader(vertShader);
      if (fragShader) gl.deleteShader(fragShader);
      if (program) gl.deleteProgram(program);
    };
  }, [speed, density, scale, contrast, brightness, interactive]);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden rounded-2xl ${className}`}
      {...props}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full pointer-events-none block"
      />
      {children && <div className="relative z-10 h-full w-full">{children}</div>}
    </div>
  );
}

export default CloudShader;
