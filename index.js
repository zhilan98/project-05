const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(require('cors')());

app.post('/process', upload.single('image'), (req, res) => {
  const method = req.body.method;
  const inputPath = req.file.path;
  const outputPath = `${inputPath}_output.jpg`;

  const python = spawn('python', [
    path.join(__dirname, '../processor/process_image.py'),
    inputPath,
    outputPath,
    method
  ]);

  python.on('close', (code) => {
    if (code === 0) {
      res.sendFile(path.resolve(outputPath), () => {
        fs.unlinkSync(inputPath);
        fs.unlinkSync(outputPath);
      });
    } else {
      res.status(500).send("处理失败");
    }
  });
});

app.listen(5000, () => console.log("✅ Server running on http://localhost:5000"));
