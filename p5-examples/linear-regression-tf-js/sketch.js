let x_vals = [];
let y_vals = [];

let m, b;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(400, 400);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function mousePressed() {
  if (
    mouseX >= 0 &&
    mouseX <= width &&
    mouseY >= 0 &&
    mouseY <= height
  ) {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);
    x_vals.push(x);
    y_vals.push(y);
  }
}

function draw() {
  background(0);
  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    point(px, py);
  }

  let lineX = [0, 1];
  // console.log(xs);
  let ys = tf.tidy(() => predict(lineX));
  let lineY = ys.dataSync();
  ys.dispose();
  // ys.print();

  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);
  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);

  if (x_vals.length > 1) {
    tf.tidy(() => {
      const y_tensor = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), y_tensor));
    });

    stroke(255, 0, 255);
    strokeWeight(2);
    line(x1, y1, x2, y2);
  }
}

function predict(x) {
  const x_tensor = tf.tensor1d(x);
  // y = mx +b
  const y = x_tensor.mul(m).add(b);
  return y;
}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}
