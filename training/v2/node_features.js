// Features from the 8-bit TF.js backbone (EXPORT_FEATURES=1 python export_tfjs.py ...), for QUANT_FEATURES in build_v2.py.
// Needs: npm install @tensorflow/tfjs-node
// usage: node feats.js MODEL_DIR OUT.bin FLIP(0/1) < paths.txt   -> float32 [n, d] features
const tf = require('@tensorflow/tfjs-node'); const fs = require('fs');
(async () => {
  const [dir, out, flip] = process.argv.slice(2);
  const m = await tf.loadGraphModel('file://' + dir + '/model.json');
  const paths = fs.readFileSync(0, 'utf8').trim().split('\n');
  const fd = fs.openSync(out, 'w'); let k = 0;
  for (const p of paths) {
    const f = tf.tidy(() => {
      let x = tf.image.resizeBilinear(tf.node.decodeImage(fs.readFileSync(p), 3), [384, 384], false, true).toFloat().expandDims(0);
      if (flip === '1') x = tf.image.flipLeftRight(x);
      return m.predict(x).dataSync();
    });
    fs.writeSync(fd, Buffer.from(new Float32Array(f).buffer));
    if (++k % 500 === 0) console.log(k, '/', paths.length);
  }
  fs.closeSync(fd);
})();
