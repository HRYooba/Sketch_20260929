// Webカメラの映像ストリームだけを扱う。エッジ抽出は renderer 側の GPU パスで行う
export class Camera {
  constructor(video) {
    this.video = video;
    this.stream = null;
  }

  get on() { return this.stream !== null; }
  get ready() { return this.on && this.video.videoWidth > 0; }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
  }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }
}
