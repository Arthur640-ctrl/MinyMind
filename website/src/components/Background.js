import React, { useRef, useEffect } from "react";
import * as THREE from "three";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const Background = () => {
  const canvasRef = useRef();

  useEffect(() => {
    // === Setup scène ===
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: true,
    });

    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio > 1 ? 2 : 1);
    renderer.toneMapping = THREE.ReinhardToneMapping;
    renderer.toneMappingExposure = 1.5;

    // === Particules ===
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 2000;

    const posArray = new Float32Array(particlesCount * 3);
    const colorArray = new Float32Array(particlesCount * 3);
    const sizeArray = new Float32Array(particlesCount);

    for (let i = 0; i < particlesCount * 3; i += 3) {
      const radius = 5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;

      posArray[i] = radius * Math.sin(phi) * Math.cos(theta);
      posArray[i + 1] = radius * Math.sin(phi) * Math.sin(theta);
      posArray[i + 2] = radius * Math.cos(phi);

      const colorChoice = Math.floor(Math.random() * 3);
      if (colorChoice === 0) {
        colorArray[i] = 0;
        colorArray[i + 1] = 1;
        colorArray[i + 2] = 1;
      } else if (colorChoice === 1) {
        colorArray[i] = 1;
        colorArray[i + 1] = 0.2;
        colorArray[i + 2] = 0.6;
      } else {
        colorArray[i] = 0.5;
        colorArray[i + 1] = 0.2;
        colorArray[i + 2] = 0.9;
      }

      sizeArray[i / 3] = Math.random() * 0.1 + 0.01;
    }

    particlesGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(posArray, 3)
    );
    particlesGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(colorArray, 3)
    );
    particlesGeometry.setAttribute(
      "size",
      new THREE.BufferAttribute(sizeArray, 1)
    );

    const particlesMaterial = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particlesMesh);

    camera.position.z = 5;

    // === Animation ===
    const animate = () => {
      requestAnimationFrame(animate);

      particlesMesh.rotation.x += 0.001;
      particlesMesh.rotation.y += 0.002;
      particlesMesh.rotation.z += 0.0005;

      const time = Date.now() * 0.001;
      const positions = particlesGeometry.attributes.position.array;

      for (let i = 0; i < positions.length; i += 3) {
        positions[i] += Math.sin(time + i) * 0.001;
        positions[i + 1] += Math.cos(time + i) * 0.001;
        positions[i + 2] += Math.sin(time + i * 0.5) * 0.001;
      }

      particlesGeometry.attributes.position.needsUpdate = true;

      renderer.render(scene, camera);
    };
    animate();

    // === Resize ===
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
    };
  }, []);

  return <canvas ref={canvasRef} style={{ position: "fixed", top: 0, left: 0, zIndex: 0, width: "100%", height: "100%" }} />;
};

export default Background;
