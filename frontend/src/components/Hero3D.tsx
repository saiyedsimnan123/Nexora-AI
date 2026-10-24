import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

function checkWebGLSupport(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(
      window.WebGLRenderingContext &&
        (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    );
  } catch {
    return false;
  }
}

export const Hero3D: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hasWebGL] = useState<boolean>(() => checkWebGLSupport());
  const [isReducedMotion, setIsReducedMotion] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handleMotionChange = (e: MediaQueryListEvent) => {
      setIsReducedMotion(e.matches);
    };
    mediaQuery.addEventListener('change', handleMotionChange);

    if (!hasWebGL || !containerRef.current) {
      return () => mediaQuery.removeEventListener('change', handleMotionChange);
    }

    const container = containerRef.current;
    let width = container.clientWidth || 500;
    let height = container.clientHeight || 500;

    // 2. Three.js Scene Setup
    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.z = 7.5;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // 3. Lighting
    const ambientLight = new THREE.AmbientLight(0x0e1329, 2.0);
    scene.add(ambientLight);

    const cyanLight = new THREE.PointLight(0x00f5d4, 4.0, 15);
    cyanLight.position.set(4, 3, 4);
    scene.add(cyanLight);

    const violetLight = new THREE.PointLight(0x8b5cf6, 3.5, 15);
    violetLight.position.set(-4, -3, 3);
    scene.add(violetLight);

    const dirLight = new THREE.DirectionalLight(0xe0e7ff, 1.2);
    dirLight.position.set(0, 5, 5);
    scene.add(dirLight);

    // 4. Central Research Intelligence Group
    const coreGroup = new THREE.Group();
    scene.add(coreGroup);

    // 4a. Inner Nucleus
    const nucleusGeo = new THREE.SphereGeometry(0.85, 32, 32);
    const nucleusMat = new THREE.MeshStandardMaterial({
      color: 0x060919,
      emissive: 0x4f46e5,
      emissiveIntensity: 0.6,
      roughness: 0.15,
      metalness: 0.9,
    });
    const nucleus = new THREE.Mesh(nucleusGeo, nucleusMat);
    coreGroup.add(nucleus);

    // 4b. Geodesic Wireframe Shell (Cyan)
    const wireGeo1 = new THREE.IcosahedronGeometry(1.35, 2);
    const wireMat1 = new THREE.MeshBasicMaterial({
      color: 0x00f5d4,
      wireframe: true,
      transparent: true,
      opacity: 0.35,
    });
    const wireMesh1 = new THREE.Mesh(wireGeo1, wireMat1);
    coreGroup.add(wireMesh1);

    // 4c. Outer Geodesic Shell (Violet)
    const wireGeo2 = new THREE.IcosahedronGeometry(1.68, 1);
    const wireMat2 = new THREE.MeshBasicMaterial({
      color: 0x8b5cf6,
      wireframe: true,
      transparent: true,
      opacity: 0.22,
    });
    const wireMesh2 = new THREE.Mesh(wireGeo2, wireMat2);
    coreGroup.add(wireMesh2);

    // 5. Orbital Rings
    const ringGroup = new THREE.Group();
    coreGroup.add(ringGroup);

    const ringMat1 = new THREE.MeshBasicMaterial({
      color: 0x00f5d4,
      transparent: true,
      opacity: 0.4,
    });
    const ring1 = new THREE.Mesh(
      new THREE.TorusGeometry(2.35, 0.012, 16, 100),
      ringMat1
    );
    ring1.rotation.x = Math.PI * 0.35;
    ring1.rotation.y = Math.PI * 0.1;
    ringGroup.add(ring1);

    const ringMat2 = new THREE.MeshBasicMaterial({
      color: 0x8b5cf6,
      transparent: true,
      opacity: 0.35,
    });
    const ring2 = new THREE.Mesh(
      new THREE.TorusGeometry(2.8, 0.012, 16, 100),
      ringMat2
    );
    ring2.rotation.x = -Math.PI * 0.25;
    ring2.rotation.y = Math.PI * 0.2;
    ringGroup.add(ring2);

    const ringMat3 = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.25,
    });
    const ring3 = new THREE.Mesh(
      new THREE.TorusGeometry(3.2, 0.01, 16, 100),
      ringMat3
    );
    ring3.rotation.x = Math.PI * 0.15;
    ring3.rotation.z = -Math.PI * 0.2;
    ringGroup.add(ring3);

    // 6. Connected Research Nodes & Line Traces
    const nodeCount = 10;
    const nodes: THREE.Mesh[] = [];
    const nodeCoords: THREE.Vector3[] = [];

    const nodeMatCyan = new THREE.MeshStandardMaterial({
      color: 0x00f5d4,
      emissive: 0x00f5d4,
      emissiveIntensity: 0.9,
      roughness: 0.1,
    });
    const nodeMatViolet = new THREE.MeshStandardMaterial({
      color: 0xc084fc,
      emissive: 0x8b5cf6,
      emissiveIntensity: 0.9,
      roughness: 0.1,
    });

    for (let i = 0; i < nodeCount; i++) {
      const radius = 2.0 + Math.random() * 1.3;
      const theta = (i / nodeCount) * Math.PI * 2 + Math.random() * 0.3;
      const phi = (Math.random() - 0.5) * Math.PI * 0.8;

      const x = radius * Math.cos(theta) * Math.cos(phi);
      const y = radius * Math.sin(phi);
      const z = radius * Math.sin(theta) * Math.cos(phi);
      const pos = new THREE.Vector3(x, y, z);
      nodeCoords.push(pos);

      const nodeMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.065 + Math.random() * 0.03, 16, 16),
        i % 2 === 0 ? nodeMatCyan : nodeMatViolet
      );
      nodeMesh.position.copy(pos);
      coreGroup.add(nodeMesh);
      nodes.push(nodeMesh);
    }

    // Connect nodes to core and adjacent nodes with evidence lines
    const linePositions: number[] = [];
    for (let i = 0; i < nodeCount; i++) {
      const pos = nodeCoords[i];
      // Line from node to center
      linePositions.push(pos.x, pos.y, pos.z, 0, 0, 0);

      // Line to neighbor node
      const nextPos = nodeCoords[(i + 1) % nodeCount];
      linePositions.push(pos.x, pos.y, pos.z, nextPos.x, nextPos.y, nextPos.z);
    }

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(linePositions, 3)
    );
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.18,
    });
    const lineSegments = new THREE.LineSegments(lineGeo, lineMat);
    coreGroup.add(lineSegments);

    // 7. Ambient Particle Field (Scientific Stardust)
    const particleCount = 450;
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      particlePositions[i] = (Math.random() - 0.5) * 12;
      particlePositions[i + 1] = (Math.random() - 0.5) * 12;
      particlePositions[i + 2] = (Math.random() - 0.5) * 8;
    }
    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(particlePositions, 3)
    );
    const particleMat = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 0.038,
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
    });
    const particleSystem = new THREE.Points(particleGeo, particleMat);
    scene.add(particleSystem);

    // 8. Mouse Tracking & Parallax
    let mouseX = 0;
    let mouseY = 0;
    let targetRotationX = 0;
    let targetRotationY = 0;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      mouseX = x * 2;
      mouseY = y * 2;
    };

    window.addEventListener('mousemove', handleMouseMove);

    // 9. Resize Handling
    const handleResize = () => {
      if (!container) return;
      width = container.clientWidth;
      height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    const resizeObserver = new ResizeObserver(() => handleResize());
    resizeObserver.observe(container);

    // 10. Animation Loop
    let animationFrameId: number;
    let lastTime = performance.now();
    const startTime = performance.now();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      if (mediaQuery.matches) {
        // Reduced motion: static rendering with fixed subtle orientation
        renderer.render(scene, camera);
        return;
      }

      const now = performance.now();
      const delta = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;
      const elapsedTime = (now - startTime) / 1000;

      // Mouse Parallax Damping
      targetRotationY = mouseX * 0.45;
      targetRotationX = mouseY * 0.35;

      coreGroup.rotation.y += (targetRotationY - coreGroup.rotation.y) * 0.04 + delta * 0.18;
      coreGroup.rotation.x += (targetRotationX - coreGroup.rotation.x) * 0.04;

      // Subtle dynamic counter-rotation on orbital rings
      ring1.rotation.z += delta * 0.15;
      ring2.rotation.z -= delta * 0.12;
      ring3.rotation.z += delta * 0.08;

      // Subtle breathing scale on nucleus
      const scale = 1 + Math.sin(elapsedTime * 1.5) * 0.03;
      nucleus.scale.set(scale, scale, scale);

      // Slow particle drift
      particleSystem.rotation.y = elapsedTime * 0.02;

      renderer.render(scene, camera);
    };

    animate();

    // 11. Cleanup
    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('mousemove', handleMouseMove);
      mediaQuery.removeEventListener('change', handleMotionChange);
      resizeObserver.disconnect();

      // Dispose Geometries & Materials
      nucleusGeo.dispose();
      nucleusMat.dispose();
      wireGeo1.dispose();
      wireMat1.dispose();
      wireGeo2.dispose();
      wireMat2.dispose();
      lineGeo.dispose();
      lineMat.dispose();
      particleGeo.dispose();
      particleMat.dispose();

      renderer.dispose();
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [hasWebGL]);

  // Graceful WebGL / Reduced Motion 2D Fallback
  if (!hasWebGL) {
    return (
      <div
        role="img"
        aria-label="Scientific Intelligence Core (2D Visualization)"
        className="relative w-full h-[460px] md:h-[540px] flex items-center justify-center p-6 select-none"
      >
        <div className="relative w-80 h-80 rounded-full border border-cyan-500/20 bg-gradient-to-br from-cyan-950/20 to-purple-950/20 flex items-center justify-center">
          <div className="absolute inset-4 rounded-full border border-dashed border-violet-500/30 animate-spin [animation-duration:40s]" />
          <div className="absolute inset-12 rounded-full border border-cyan-400/25 animate-spin [animation-duration:25s] [animation-direction:reverse]" />
          <div className="w-28 h-28 rounded-full bg-gradient-to-br from-cyan-400/20 via-indigo-600/30 to-purple-600/20 backdrop-blur-md border border-cyan-400/40 flex items-center justify-center shadow-[0_0_40px_rgba(0,245,212,0.25)]">
            <div className="w-12 h-12 rounded-full bg-cyan-400/80 shadow-[0_0_20px_#00f5d4]" />
          </div>

          {/* Research node dots */}
          <div className="absolute top-4 left-1/4 w-3 h-3 rounded-full bg-cyan-300 shadow-[0_0_10px_#00f5d4]" />
          <div className="absolute bottom-6 right-1/4 w-3.5 h-3.5 rounded-full bg-violet-400 shadow-[0_0_12px_#8b5cf6]" />
          <div className="absolute top-1/2 right-3 w-2.5 h-2.5 rounded-full bg-blue-400 shadow-[0_0_8px_#38bdf8]" />
          <div className="absolute bottom-1/3 left-4 w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_10px_#00f5d4]" />
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-[460px] md:h-[540px] lg:h-[580px] flex items-center justify-center overflow-hidden">
      {/* Background radial glow behind 3D core */}
      <div className="absolute w-80 h-80 rounded-full bg-cyan-500/10 blur-[100px] pointer-events-none" />
      <div className="absolute w-72 h-72 rounded-full bg-violet-600/15 blur-[90px] translate-x-12 translate-y-10 pointer-events-none" />

      {/* 3D Canvas Container */}
      <div
        ref={containerRef}
        aria-label="Interactive 3D Scientific Intelligence Core"
        role="img"
        className="w-full h-full cursor-grab active:cursor-grabbing"
      />

      {/* Reduced-Motion Badge if applicable */}
      {isReducedMotion && (
        <div className="absolute bottom-3 right-3 text-[11px] font-mono text-slate-400 bg-slate-900/80 px-2.5 py-1 rounded-full border border-slate-700/50 backdrop-blur-sm pointer-events-none">
          Reduced Motion Active
        </div>
      )}
    </div>
  );
};
