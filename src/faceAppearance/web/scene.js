import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const VIEW_DIRECTIONS = {
  front: new THREE.Vector3(0, 0, 1),
  left: new THREE.Vector3(1, 0, 0),
  right: new THREE.Vector3(-1, 0, 0),
  back: new THREE.Vector3(0, 0, -1),
};

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function finiteArray(values, name, stride) {
  assert(Array.isArray(values) && values.length > 0 && values.length % stride === 0
    && values.every(Number.isFinite), `Invalid preview ${name}.`);
}

function disposeAsset(asset) {
  if (!asset) return;
  asset.mesh?.removeFromParent();
  asset.geometry?.dispose();
  for (const material of asset.materials) material.dispose();
  for (const texture of asset.textures) texture.dispose();
  for (const url of asset.urls) URL.revokeObjectURL(url);
  asset.mesh = null;
  asset.geometry = null;
  asset.materials.length = 0;
  asset.textures.clear();
  asset.urls.clear();
  asset.modes = {};
}

/** A static asset viewer: original coordinates and source vertex identities stay intact. */
export class FaceViewer {
  constructor(container, { onError = () => {} } = {}) {
    this.container = container;
    this.onError = onError;
    this.disposed = false;
    this.generation = 0;
    this.asset = null;
    this.pending = new Set();
    this.mode = 'textured';
    this.center = new THREE.Vector3();
    this.radius = 0.2;
    this.fitDistance = 1;
    this.hasBounds = false;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.001, 100);
    this.camera.position.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setClearColor(0x101719, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1;
    const canvas = this.renderer.domElement;
    canvas.setAttribute('aria-label', '可旋转的三维头部资产预览');
    canvas.setAttribute('role', 'img');
    canvas.tabIndex = 0;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    canvas.style.touchAction = 'none';
    this.container.appendChild(canvas);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.rotateSpeed = 0.6;
    this.controls.zoomSpeed = 0.8;
    this.controls.panSpeed = 0.6;
    this.controls.autoRotateSpeed = 1;
    this.controls.minPolarAngle = 0.06;
    this.controls.maxPolarAngle = Math.PI - 0.06;

    this.scene.add(new THREE.HemisphereLight(0xe9f2ff, 0x555052, 1.75));
    this.lightRig = new THREE.Group();
    const lights = [
      { color: 0xfff4e9, intensity: 3.0, position: [-2.5, 3.0, 4.0] },
      { color: 0xdcecff, intensity: 1.8, position: [3.0, 0.5, 2.0] },
      { color: 0xe4faff, intensity: 2.2, position: [0.5, 2.5, -3.0] },
    ];
    for (const descriptor of lights) {
      const light = new THREE.DirectionalLight(descriptor.color, descriptor.intensity);
      light.position.fromArray(descriptor.position);
      this.lightRig.add(light, light.target);
    }
    this.scene.add(this.lightRig);

    this.onContextLost = (event) => {
      event.preventDefault();
      this.contextLost = true;
      this.onError(new Error('三维预览的 WebGL 上下文已丢失；请刷新页面恢复。'));
    };
    this.onContextRestored = () => { this.contextLost = false; };
    canvas.addEventListener('webglcontextlost', this.onContextLost);
    canvas.addEventListener('webglcontextrestored', this.onContextRestored);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.lastFrameTime = null;
    this.renderer.setAnimationLoop((time) => {
      if (this.disposed || this.contextLost) return;
      const delta = this.lastFrameTime === null ? 0 : Math.min((time - this.lastFrameTime) / 1000, 0.1);
      this.lastFrameTime = time;
      this.controls.update(delta);
      this.renderer.render(this.scene, this.camera);
    });
  }

  _assertCurrent(generation) {
    if (this.disposed || generation !== this.generation) {
      throw new DOMException('Preview load was superseded.', 'AbortError');
    }
  }

  async _texture(descriptor, fetchTexture, asset, generation) {
    const blob = await fetchTexture(descriptor.url);
    this._assertCurrent(generation);
    assert(blob instanceof Blob, 'Texture loader must return a Blob.');
    const url = URL.createObjectURL(blob);
    asset.urls.add(url);
    let texture;
    try {
      // TextureLoader decodes an HTMLImageElement; flipY is respected (unlike ImageBitmap).
      texture = await new THREE.TextureLoader().loadAsync(url);
      asset.textures.add(texture);
      this._assertCurrent(generation);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.flipY = false; // The validated render arrays already use V-down UVs.
      texture.anisotropy = Math.min(this.renderer.capabilities.getMaxAnisotropy(), 8);
      texture.needsUpdate = true;
      return texture;
    } finally {
      URL.revokeObjectURL(url);
      asset.urls.delete(url);
    }
  }

  _alphaTexture(texture, asset) {
    // Three's alphaMap samples GREEN, not the PNG alpha channel. Copy alpha
    // into RGB to keep brows/lashes cut out when their color map is hidden.
    const source = texture.image;
    const canvas = document.createElement('canvas');
    canvas.width = source.naturalWidth || source.width;
    canvas.height = source.naturalHeight || source.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    assert(context, 'Cannot decode cutout texture.');
    context.drawImage(source, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
    for (let offset = 0; offset < pixels.data.length; offset += 4) {
      const alpha = pixels.data[offset + 3];
      pixels.data[offset] = alpha;
      pixels.data[offset + 1] = alpha;
      pixels.data[offset + 2] = alpha;
      pixels.data[offset + 3] = 255;
    }
    context.putImageData(pixels, 0, 0);
    const alphaMap = new THREE.CanvasTexture(canvas);
    alphaMap.flipY = false;
    alphaMap.colorSpace = THREE.NoColorSpace;
    alphaMap.anisotropy = texture.anisotropy;
    asset.textures.add(alphaMap);
    return alphaMap;
  }

  async load(preview, fetchTexture) {
    assert(!this.disposed, 'Viewer is disposed.');
    const generation = ++this.generation;
    const asset = { geometry: null, mesh: null, materials: [], textures: new Set(), urls: new Set(), modes: {} };
    this.pending.add(asset);
    let committed = false;
    try {
      const { mesh, materials } = preview;
      assert(mesh && Array.isArray(materials) && materials.length > 0, 'Missing preview mesh or materials.');
      finiteArray(mesh.positions, 'positions', 3);
      finiteArray(mesh.normals, 'normals', 3);
      finiteArray(mesh.uvs, 'UVs', 2);
      const vertexCount = mesh.positions.length / 3;
      assert(mesh.normals.length === mesh.positions.length && mesh.uvs.length / 2 === vertexCount,
        'Preview attribute counts differ.');
      assert(Array.isArray(mesh.indices) && mesh.indices.length > 0 && mesh.indices.length % 3 === 0
        && mesh.indices.every((value) => Number.isInteger(value) && value >= 0 && value < vertexCount),
      'Invalid preview indices.');
      assert(Array.isArray(mesh.sourceVertexIds) && mesh.sourceVertexIds.length === vertexCount
        && mesh.sourceVertexIds.every((value) => Number.isInteger(value) && value >= 0 && value <= 0xffffffff),
      'Invalid source vertex mapping.');
      const geometry = asset.geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(mesh.positions, 3));
      // Normals were averaged over SOURCE vertices before UV seam expansion.
      geometry.setAttribute('normal', new THREE.Float32BufferAttribute(mesh.normals, 3));
      geometry.setAttribute('uv', new THREE.Float32BufferAttribute(mesh.uvs, 2));
      geometry.setAttribute('sourceVertexId', new THREE.Uint32BufferAttribute(mesh.sourceVertexIds, 1));
      geometry.setIndex(new THREE.Uint32BufferAttribute(mesh.indices, 1));
      assert(Array.isArray(mesh.groups) && mesh.groups.length > 0, 'Missing preview material groups.');
      let groupEnd = 0;
      for (const group of mesh.groups) {
        assert(Number.isInteger(group.start) && group.start === groupEnd
          && Number.isInteger(group.count) && group.count > 0 && group.count % 3 === 0
          && Number.isInteger(group.materialIndex) && group.materialIndex >= 0 && group.materialIndex < materials.length,
        'Invalid preview material group.');
        geometry.addGroup(group.start, group.count, group.materialIndex);
        groupEnd += group.count;
      }
      assert(groupEnd === mesh.indices.length, 'Material groups do not cover the mesh.');
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();

      asset.modes = { textured: [], clay: [], wireframe: [] };
      // Sequential loading keeps cleanup deterministic on rejection or a newer request.
      for (let index = 0; index < materials.length; index += 1) {
        const descriptor = materials[index];
        assert(descriptor.index === index, 'Preview materials must follow their declared index.');
        assert(['OPAQUE', 'MASK'].includes(descriptor.alphaMode), 'Unsupported preview alpha mode.');
        const map = descriptor.texture ? await this._texture(descriptor.texture, fetchTexture, asset, generation) : null;
        this._assertCurrent(generation);
        const cutout = descriptor.alphaMode === 'MASK';
        const factor = descriptor.baseColorFactor || [1, 1, 1, 1];
        const base = {
          name: descriptor.name,
          color: new THREE.Color().setRGB(factor[0], factor[1], factor[2], THREE.LinearSRGBColorSpace),
          opacity: factor[3] ?? 1,
          roughness: descriptor.roughnessFactor ?? 0.6,
          metalness: descriptor.metallicFactor ?? 0,
          map,
          alphaTest: cutout ? (descriptor.alphaCutoff ?? 0.5) : 0,
          transparent: false,
          side: cutout ? THREE.DoubleSide : THREE.FrontSide,
        };
        const textured = new THREE.MeshStandardMaterial(base);
        asset.materials.push(textured);
        asset.modes.textured.push(textured);
        const alphaMap = cutout && map ? this._alphaTexture(map, asset) : null;
        const clay = new THREE.MeshStandardMaterial({
          ...base, name: `${descriptor.name}:clay`, color: 0xb8c4c5,
          map: null, alphaMap, roughness: 0.85, metalness: 0,
        });
        const wireframe = new THREE.MeshStandardMaterial({
          ...base, name: `${descriptor.name}:wireframe`, color: 0x84cbbc,
          map: null, alphaMap, roughness: 1, metalness: 0, wireframe: true,
        });
        asset.materials.push(clay, wireframe);
        asset.modes.clay.push(clay);
        asset.modes.wireframe.push(wireframe);
      }
      this._assertCurrent(generation);
      const nextMesh = asset.mesh = new THREE.Mesh(geometry, asset.modes[this.mode]);
      nextMesh.name = 'Validated client asset';
      nextMesh.userData.templateId = preview.templateId;
      nextMesh.userData.units = 'meters';
      disposeAsset(this.asset);
      this.asset = asset;
      this.scene.add(nextMesh);
      geometry.boundingBox.getCenter(this.center);
      this.radius = Math.max(geometry.boundingSphere.radius, 0.001);
      this.hasBounds = true;
      this.lightRig.position.copy(this.center);
      this.lightRig.scale.setScalar(this.radius);
      this.camera.near = Math.max(this.radius / 100, 0.00001);
      this.camera.far = Math.max(this.radius * 100, 10);
      this.controls.minDistance = this.radius * 0.55;
      this.controls.maxDistance = this.radius * 15;
      this.fitDistance = this._fitDistance();
      this.camera.updateProjectionMatrix();
      this.reset();
      committed = true;
      return { vertices: vertexCount, triangles: mesh.indices.length / 3 };
    } finally {
      this.pending.delete(asset);
      if (!committed) disposeAsset(asset);
    }
  }

  _fitDistance() {
    const halfVertical = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const halfHorizontal = Math.atan(Math.tan(halfVertical) * this.camera.aspect);
    return this.radius / Math.sin(Math.min(halfVertical, halfHorizontal)) * 1.08;
  }

  setMode(mode) {
    assert(['textured', 'clay', 'wireframe'].includes(mode), 'Unknown preview mode.');
    this.mode = mode;
    if (this.asset) this.asset.mesh.material = this.asset.modes[mode];
  }

  setView(view) {
    assert(Object.hasOwn(VIEW_DIRECTIONS, view), 'Unknown camera view.');
    // Clear any inertial motion before applying an exact inspection direction.
    const damping = this.controls.enableDamping;
    const autoRotate = this.controls.autoRotate;
    this.controls.enableDamping = false;
    this.controls.autoRotate = false;
    this.controls.update();
    this.controls.target.copy(this.center);
    this.camera.position.copy(this.center).addScaledVector(VIEW_DIRECTIONS[view], this.fitDistance);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(this.center);
    this.controls.update();
    this.controls.enableDamping = damping;
    this.controls.autoRotate = autoRotate;
  }

  setAutoRotate(enabled) {
    this.controls.autoRotate = Boolean(enabled);
  }

  setExposure(value) {
    assert(Number.isFinite(value) && value > 0, 'Exposure must be positive.');
    this.renderer.toneMappingExposure = Math.min(Math.max(value, 0.1), 4);
  }

  reset() {
    this.setView('front');
  }

  clear() {
    if (this.disposed) return;
    this.generation += 1;
    disposeAsset(this.asset);
    this.asset = null;
    this.hasBounds = false;
    for (const asset of this.pending) disposeAsset(asset);
    // The scene is reusable. Pending decoders check the generation before they
    // install any asset, and their finally blocks release late-arriving data.
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    if (this.disposed) return;
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    const previousFit = this.fitDistance;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    if (this.hasBounds) {
      this.fitDistance = this._fitDistance();
      this.camera.position.sub(this.controls.target).multiplyScalar(this.fitDistance / previousFit)
        .add(this.controls.target);
    }
  }

  screenshot() {
    assert(!this.disposed && !this.contextLost, 'Three-dimensional preview is unavailable.');
    // Render and read synchronously, before WebGL clears its drawing buffer.
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.generation += 1;
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    disposeAsset(this.asset);
    this.asset = null;
    // Release resources already acquired by pending requests immediately. A
    // decoder settling later is rejected and disposes its newly created texture.
    for (const asset of this.pending) disposeAsset(asset);
    const canvas = this.renderer.domElement;
    canvas.removeEventListener('webglcontextlost', this.onContextLost);
    canvas.removeEventListener('webglcontextrestored', this.onContextRestored);
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    canvas.remove();
  }
}
