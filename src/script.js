import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import * as dat from "lil-gui";
import { RectAreaLightHelper } from "three/examples/jsm/helpers/RectAreaLightHelper.js";

/**
 * Webcam Setup
 */
async function setupWebcam() {
  const video = document.createElement("video");
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });

  video.play();
  return video;
}

/**
 * Base
 */
// Debug
const gui = new dat.GUI();

// Canvas
const canvas = document.querySelector("canvas.webgl");

// Scene
const scene = new THREE.Scene();

/**
 * Lights
 */

// AmbientLight is used to simulate light bouncing off objects.
const ambientLight = new THREE.AmbientLight(0xffffff, 0.0);
scene.add(ambientLight);
gui
  .add(ambientLight, "intensity")
  .min(0)
  .max(1)
  .step(0.0001)
  .name("Ambient Intensity");

// DirectionalLight will have a sun-like effect in a parallel direction.
const directionalLight = new THREE.DirectionalLight(0x00fffc, 0.0);
directionalLight.position.set(1, 0.25, 0);
scene.add(directionalLight);
gui
  .add(directionalLight, "intensity")
  .min(0)
  .max(1)
  .step(0.0001)
  .name("Directional Int.");

// PointLight looks like a candle and has a small point that radiates out.
const pointLight = new THREE.PointLight(0xff9000, 1.0, 10, 6);
pointLight.position.set(1, -0.5, 1);
scene.add(pointLight);
gui.add(pointLight, "intensity").min(0).max(5).step(0.001).name("Point Int.");

// RectAreaLight is a mix of directional and diffused light.
const rectAreaLight = new THREE.RectAreaLight(0x4e00ff, 3, 1, 1);
const epsilon = 0.01;
rectAreaLight.position.set(0, 0, -epsilon);
rectAreaLight.scale.set(4 / 3 + 0.2, 1.2, 1); // Slightly larger than the screen
rectAreaLight.lookAt(new THREE.Vector3());
scene.add(rectAreaLight);
gui
  .add(rectAreaLight, "intensity")
  .min(0)
  .max(10)
  .step(0.001)
  .name("RectArea Int.");

// Helpers
const rectAreaLightHelper = new RectAreaLightHelper(rectAreaLight);
scene.add(rectAreaLightHelper);

/**
 * Objects
 */
// Material
const material = new THREE.MeshStandardMaterial();
material.roughness = 0.4;

// Objects

const plane = new THREE.Mesh(new THREE.PlaneGeometry(5, 5), material);
plane.rotation.x = -Math.PI * 0.5;
plane.position.y = -0.65;

scene.add(plane);

/**
 * Sizes
 */
const sizes = {
  width: window.innerWidth,
  height: window.innerHeight,
};

window.addEventListener("resize", () => {
  // Update sizes
  sizes.width = window.innerWidth;
  sizes.height = window.innerHeight;

  // Update camera
  camera.aspect = sizes.width / sizes.height;
  camera.updateProjectionMatrix();

  // Update renderer
  renderer.setSize(sizes.width, sizes.height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

/**
 * Camera
 */
// Base camera
const camera = new THREE.PerspectiveCamera(
  75,
  sizes.width / sizes.height,
  0.1,
  100
);
camera.position.x = 0;
camera.position.y = 0;
camera.position.z = 1.25;
scene.add(camera);

// Controls
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.minPolarAngle = Math.PI / 2.01; // Slightly above the horizon

/**
 * Renderer
 */
const renderer = new THREE.WebGLRenderer({
  canvas: canvas,
});
renderer.setSize(sizes.width, sizes.height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

/**
 * Animate
 */
const clock = new THREE.Clock();
let videoTexture = null;

const tick = () => {
  const elapsedTime = clock.getElapsedTime();

  // Update objects
  // sphere.rotation.y = 0.1 * elapsedTime;
  // sphere.rotation.x = 0.15 * elapsedTime;

  // Update video texture
  if (videoTexture) {
    videoTexture.needsUpdate = true;
  }

  // Update controls
  controls.update();

  // Render
  renderer.render(scene, camera);

  // Call tick again on the next frame
  window.requestAnimationFrame(tick);
};

/**
 * Initialize
 */
async function init() {
  // Set up the webcam
  const video = await setupWebcam();

  // Create a texture from the video element
  videoTexture = new THREE.VideoTexture(video);
  videoTexture.minFilter = THREE.LinearFilter;
  videoTexture.magFilter = THREE.LinearFilter;
  videoTexture.format = THREE.RGBFormat;

  // Create a plane geometry and material using the video texture
  const screenGeometry = new THREE.PlaneGeometry(4 / 3, 1);
  const screenMaterial = new THREE.MeshBasicMaterial({ map: videoTexture });
  const screen = new THREE.Mesh(screenGeometry, screenMaterial);

  // Position the screen in the scene
  screen.position.set(0, 0, 0);
  screen.scale.x = -1;
  scene.add(screen);

  // Start the animation loop
  tick();
}

init();
