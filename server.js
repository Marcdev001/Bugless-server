const express = require('express');
const mongoose = require('mongoose');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs/promises');
const helmet = require('helmet');
const dotenv = require('dotenv');
const { errorHandler } = require('./middleware/errorHandler.js');
const Project = require('./models/Project.js');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(helmet());
app.use(cors());
app.use(express.json());

const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedExtensions = ['.html', '.css', '.js', '.jsx', '.ts', '.tsx', '.py'];
    const fileExtension = path.extname(file.originalname).toLowerCase();
    if (allowedExtensions.includes(fileExtension)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only supported code files are allowed.'));
    }
  },
  limits: {
    fileSize: 5 * 1024 * 1024
  }
});

app.post('/api/analyze', upload.array('files', 20), async (req, res, next) => {
  try {
    const files = req.files.map(file => ({
      name: file.originalname,
      path: file.path
    }));

    const { analyzeCode } = require('./services/codeAnalyzer');
    const analysisResults = await analyzeCode(files);

    res.json({ analysisResults });
  } catch (error) {
    console.error('Error in /api/analyze:', error);
    next(error);
  }
});

app.use(errorHandler);

mongoose.connect(process.env.MONGODB_URI)
  .then(() => {
    console.log('Connected to MongoDB');
    app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT}`);
    });
  })
  .catch((err) => console.error('MongoDB connection error:', err));