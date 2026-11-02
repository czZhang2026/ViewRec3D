// Missing figures remain intentional placeholders; replace them by adding the named files.
for (const slot of document.querySelectorAll('[data-image]')) {
  const image = new Image();
  image.alt = slot.dataset.alt;
  image.className = 'asset-image';
  image.onload = () => slot.replaceWith(image);
  image.src = slot.dataset.image;
}

const $ = (id) => document.getElementById(id);
const status = (message) => { $('demo-status').textContent = message; };
for (const [group, values] of [['rotation', [1, 0, 0, 0, 1, 0, 0, 0, 1]], ['translation', [0, 0, 0]]]) {
  values.forEach((value, index) => {
    const input = document.createElement('input');
    Object.assign(input, { type: 'number', step: 'any', value, required: true, id: `${group}-${index}` });
    input.setAttribute('aria-label', group === 'rotation' ? `Rotation row ${Math.floor(index / 3) + 1}, column ${index % 3 + 1}` : `Translation ${['x', 'y', 'z'][index]}`);
    $(`${group}-inputs`).append(input);
  });
}

let scenes = [], active, renderer, camera, world, mesh, THREE, SplatMesh;
let busy = false;
const read = (group, count) => Array.from({ length: count }, (_, i) => Number($( `${group}-${i}`).value));
const write = (group, values) => values.forEach((value, i) => { $(`${group}-${i}`).value = value; });
function validatePose(pose) {
  const { rot: r, trans: t, fovY } = pose;
  if (!Array.isArray(r) || r.length !== 9 || !Array.isArray(t) || t.length !== 3 || ![...r, ...t, fovY].every(Number.isFinite)) throw new Error('Use nine finite rotation values, three translation values, and a finite field of view.');
  if (fovY < 1 || fovY > 170) throw new Error('Vertical field of view must be between 1 and 170 degrees.');
  const determinant = r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    const dot = r[i*3]*r[j*3]+r[i*3+1]*r[j*3+1]+r[i*3+2]*r[j*3+2];
    if (Math.abs(dot - (i === j ? 1 : 0)) > .01) throw new Error('Rotation rows must be orthogonal unit vectors.');
  }
  if (Math.abs(determinant - 1) > .01) throw new Error('Rotation must have determinant +1.');
}
function currentPose() {
  return { rot: read('rotation', 9), trans: read('translation', 3), fovY: Number($('fov').value) };
}
function setInitialPose() {
  write('rotation', active.initialPose.rot);
  write('translation', active.initialPose.trans);
  $('fov').value = active.initialPose.fovY;
  $('axes').value = active.cameraAxes;
}
function applyPose(pose) {
  validatePose(pose);
  const r = pose.rot, t = pose.trans;
  const transform = new THREE.Matrix4().set(r[0],r[1],r[2],t[0], r[3],r[4],r[5],t[1], r[6],r[7],r[8],t[2], 0,0,0,1);
  // c2w maps camera coordinates into world coordinates. Convert camera basis only.
  if ($('axes').value === 'opencv') transform.multiply(new THREE.Matrix4().makeScale(1, -1, -1));
  camera.position.setFromMatrixPosition(transform);
  camera.quaternion.setFromRotationMatrix(transform);
  camera.fov = pose.fovY;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
}
async function initializeRenderer() {
  if (renderer) return;
  THREE = await import('three');
  const spark = await import('@sparkjsdev/spark');
  SplatMesh = spark.SplatMesh;
  world = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, 1, .01, 10000);
  const next = new THREE.WebGLRenderer({ antialias: false });
  next.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  next.setClearColor(0xeef3f1);
  world.add(new spark.SparkRenderer({ renderer: next }));
  renderer = next;
  $('viewer').append(renderer.domElement);
  new ResizeObserver(() => {
    const { width, height } = $('viewer').getBoundingClientRect();
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }).observe($('viewer'));
  renderer.setAnimationLoop(() => { if (mesh) renderer.render(world, camera); });
}
function updateReference(scene) {
  const slot = $('optimal-media');
  slot.replaceChildren();
  slot.classList.remove('has-image');
  const label = document.createElement('strong');
  label.textContent = 'Expert reference image pending';
  slot.append(label);
  if (scene.optimalImage) {
    const image = new Image();
    image.alt = `${scene.label}: optimal viewpoint reference`;
    image.onload = () => { if (active === scene) { slot.replaceChildren(image); slot.classList.add('has-image'); } };
    image.src = scene.optimalImage;
  }
  $('optimal-pose').textContent = scene.optimalPose
    ? `Optimal c2w pose\nrot: ${JSON.stringify(scene.optimalPose.rot)}\ntrans: ${JSON.stringify(scene.optimalPose.trans)}\nVertical FoV: ${scene.optimalPose.fovY}°`
    : 'Optimal c2w pose: awaiting scene metadata.';
}
async function renderView() {
  if (busy || !active) return;
  busy = true;
  $('render-button').disabled = $('scene-select').disabled = $('reset-pose').disabled = true;
  let candidate;
  try {
    const pose = currentPose();
    validatePose(pose);
    status('Loading renderer and scene…');
    await initializeRenderer();
    if (!mesh || mesh.userData.source !== active.ply) {
      candidate = new SplatMesh({ url: active.ply });
      await candidate.initialized;
      if (mesh) { world.remove(mesh); mesh.dispose(); }
      mesh = candidate;
      candidate = undefined;
      mesh.userData.source = active.ply;
      world.add(mesh);
    }
    applyPose(pose);
    $('viewer-empty').hidden = true;
    status('Scene loaded. Rendering at the entered camera-to-world pose.');
  } catch (error) {
    if (candidate) candidate.dispose();
    status(`Unable to render: ${error.message} Check the PLY path, network access, and WebGL2 support.`);
  } finally {
    busy = false;
    $('render-button').disabled = $('scene-select').disabled = $('reset-pose').disabled = false;
  }
}
$('pose-form').addEventListener('submit', (event) => { event.preventDefault(); renderView(); });
$('reset-pose').addEventListener('click', () => { setInitialPose(); renderView(); });
$('scene-select').addEventListener('change', () => {
  active = scenes[Number($('scene-select').value)];
  setInitialPose(); updateReference(active);
  if (mesh) { world.remove(mesh); mesh.dispose(); mesh = undefined; renderer.clear(); }
  $('viewer-empty').hidden = false;
  status('Scene selected. Enter a pose and select Render viewpoint.');
});
try {
  const response = await fetch('docs/scenes.json');
  if (!response.ok) throw new Error(`Scene manifest returned HTTP ${response.status}.`);
  scenes = await response.json();
  if (!Array.isArray(scenes)) throw new Error('Scene manifest must be an array.');
  for (const scene of scenes) {
    if (!scene.label || !scene.ply || !['opencv', 'opengl'].includes(scene.cameraAxes)) throw new Error('Each scene needs a label, PLY path, and cameraAxes.');
    validatePose(scene.initialPose || {});
    if (scene.optimalPose) validatePose(scene.optimalPose);
  }
  if (scenes.length) {
    $('scene-select').replaceChildren(...scenes.map((scene, i) => new Option(scene.label, i)));
    active = scenes[0]; setInitialPose(); updateReference(active);
    $('render-button').disabled = $('reset-pose').disabled = false;
    status('Select a scene, enter a c2w pose, and render a viewpoint.');
  }
} catch (error) {
  status(`Scene list unavailable: ${error.message} Preview this page over HTTP rather than opening it as a local file.`);
}
